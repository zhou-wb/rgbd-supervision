import prop_ideal
import torch

prop_dist = 0.0044
wavelength = 5.177e-07
feature_size = (6.4e-06, 6.4e-06)
F_aperture = 0.5
prop_dists_from_wrp = [-0.0044, 0.0, 0.0037999999999999987]

ASM_prop = prop_ideal.SerialProp(prop_dist, wavelength, feature_size,
                                 'ASM', F_aperture, prop_dists_from_wrp,
                                 dim=1)


input_phase = torch.load("phases_images_masks/NYU_labeled_10samples/phase/NYU_labeled_6.pt")

output_field = ASM_prop(input_phase)

output_amp = output_field.abs()

pass