"""
Neural 3D holography: Learning accurate wave propagation models for 3D holographic virtual and augmented reality displays

Suyeon Choi*, Manu Gopakumar*, Yifan Peng, Jonghyun Kim, Gordon Wetzstein

This is the main executive script used for the phase generation using SGD.
This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from Stanford).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.
-----

$ python main.py --lr=0.01 --num_iters=10000

"""
from tqdm import tqdm
from algorithms import gradient_descent
import image_loader as loaders
import numpy as np

import os
import torch
from torch.utils.tensorboard import SummaryWriter
import imageio
import configargparse
# import prop_physical
import prop_ideal
import utils
import torch.nn as nn
# import params
    

def main():
    # Command line argument processing / Parameters
    torch.set_default_dtype(torch.float32)
    p = configargparse.ArgumentParser()
    opt = p.parse_args()
    
    # propagation related
    opt.prop_dist = 0.0044
    # opt.prop_dists_from_wrp = [-0.0044, -0.0032000000000000006, -0.0024000000000000002, -0.0010000000000000005, 0.0, 0.0013, 0.0028000000000000004, 0.0037999999999999987]
    # opt.virtual_depth_planes = [0.0, 0.08417508417508479, 0.14124293785310726, 0.24299599771297942, 0.3171856978085348, 0.4155730533683304, 0.5319148936170226, 0.6112104949314254]
    opt.prop_dists_from_wrp = [-0.0044, 0.0, 0.0037999999999999987]
    opt.virtual_depth_planes = [0, 0.5, 1]
    opt.wavelength = 5.177e-07
    opt.feature_size = (6.4e-06, 6.4e-06)
    opt.F_aperture = 0.5
    
    # path related
    # input
    root_dir = '/media/datadrive/rgbd-scenes-v2/imgs'
    scene_list = ['scene_01', 'scene_02', 'scene_03', 'scene_04', 'scene_05', 'scene_06', 'scene_07', 
                'scene_08', 'scene_09', 'scene_10', 'scene_11', 'scene_12', 'scene_13', 'scene_14']
    dir_list = [os.path.join(root_dir, scene_name) for scene_name in scene_list]
    opt.data_path = dir_list
    opt.shuffle = True
    opt.num_of_samples = 1000
    # output
    opt.output_type = 'tensor' # chose from 'image' or 'tensor'
    opt.out_path = './phases_images_masks'
    run_id = 'Washington_scene_v2_1000samples'
    
    # image resolution related
    opt.channel = 1
    opt.image_res = (480, 640) # (1080, 1920)
    opt.roi_res = (480, 640) # (960, 1680)
    opt.slm_res = (480, 640) # (1080, 1920)
    
    # optimizer related
    opt.num_iters = 1000
    opt.loss_fn = nn.functional.mse_loss
    opt.lr = 0.01
    opt.init_phase_range = 1.0
    
    
    
    dev = torch.device('cuda')

    
    # path to save out optimized phases
    out_path = os.path.join(opt.out_path, run_id)
    print(f'  - out_path: {out_path}')

    # Tensorboard
    # summaries_dir = os.path.join(out_path, 'summaries')
    # utils.cond_mkdir(summaries_dir)
    # writer = SummaryWriter(summaries_dir)

    # Propagations
    ASM_prop = prop_ideal.SerialProp(opt.prop_dist, opt.wavelength, opt.feature_size,
                                         'ASM', opt.F_aperture, opt.prop_dists_from_wrp,
                                         dim=1)

    # Loader
    if ',' in opt.data_path:
        opt.data_path = opt.data_path.split(',')
    img_loader = loaders.TargetLoader(opt.data_path, channel=opt.channel,
                                      image_res=opt.image_res, roi_res=opt.roi_res,
                                      crop_to_roi=False, shuffle=opt.shuffle,
                                      virtual_depth_planes=opt.virtual_depth_planes,
                                      )

    for i, target in tqdm(enumerate(img_loader), total=opt.num_of_samples):
        
        if i >= opt.num_of_samples:
            break
        
        target_amp, target_mask, target_idx = target
        target_amp = target_amp.to(dev).detach()
        target_mask = target_mask.to(dev).detach()
        # if len(target_amp.shape) < 4:
        #     target_amp = target_amp.unsqueeze(0)

        print(f'  - run phase optimization for {target_idx}th image ...')

        # initial slm phase
        init_phase = (opt.init_phase_range * (-0.5 + 1.0 * torch.rand(1, 1, *opt.slm_res))).to(dev)

        # run algorithm
        results = gradient_descent(init_phase, target_amp, target_mask,
                                   forward_prop=ASM_prop, num_iters=opt.num_iters, roi_res=opt.roi_res,
                                   loss_fn=opt.loss_fn, lr=opt.lr,
                                   out_path_idx=f'{opt.out_path}_{target_idx}',
                                   citl=False, camera_prop=None,
                                #    writer=writer,
                                   )

        # optimized slm phase
        final_phase = results['final_phase']

        # encoding for SLM & save it out
        phase_out = utils.phasemap_8bit(final_phase)

        for i in ['phase', 'image', 'mask']:
            if not os.path.exists(os.path.join(out_path, i)):
                os.makedirs(os.path.join(out_path, i))
        
        
        if opt.output_type == 'image':
            # save phase_out as image
            phase_out_path = os.path.join(out_path, 'phase', f'{target_idx}.png')
            imageio.imwrite(phase_out_path, phase_out)
            # save images as image
            image_out_path = os.path.join(out_path, 'image', f'{target_idx}.png')
            imageio.imwrite(image_out_path, target_amp.squeeze().cpu())
            # save masks as #target_planes images
            for i, mask_i in enumerate(target_mask.squeeze()):
                mask_out_path = os.path.join(out_path, 'mask', f'{target_idx}-{i}.png')
                imageio.imwrite(mask_out_path, mask_i.cpu())
        elif opt.output_type == 'tensor':
            # save final_phase as pytorch tensor for reload
            phase_out_path = os.path.join(out_path, 'phase', f'{target_idx}.pt')
            torch.save(final_phase, phase_out_path)
            # save images as pytorch tensor
            image_out_path = os.path.join(out_path, 'image', f'{target_idx}.pt')
            torch.save(target_amp, image_out_path)
            # save masks as pytorch tensor
            mask_out_path = os.path.join(out_path, 'mask', f'{target_idx}.pt')
            torch.save(target_mask, mask_out_path)
            
        

if __name__ == "__main__":
    main()
