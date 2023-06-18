import os, shutil, random
from image_loader import get_image_filenames
from tqdm import tqdm

# root_dir = 'phases_images_masks/NYU_labeled_10samples' # 'phases_images_masks/Washington_scene_v2_1000samples'
root_dir = '/media/datadrive/img_mask_phase/FlyingThings3D'
target_dir = root_dir
shuffle = True

for i in ['train', 'test']:
    for j in ['image', 'mask', 'phase']:
        if not os.path.exists(os.path.join(target_dir, i, j)):
            os.makedirs(os.path.join(target_dir, i, j))

image_names = get_image_filenames(dir = os.path.join(root_dir, 'image'))
mask_names = get_image_filenames(dir = os.path.join(root_dir, 'mask'))
phase_names = get_image_filenames(dir = os.path.join(root_dir, 'phase'))
for names in [image_names, mask_names, phase_names]:
    names.sort()

imgnum = len(image_names)
assert(len(mask_names) == imgnum)
assert(len(phase_names) == imgnum)
order = [i for i in range(imgnum)]

if shuffle:
    random.shuffle(order)
split_idx = int(0.8*len(order))
train_list = order[:split_idx]
test_list = order[split_idx:]

print('Generating Training Set...')
for i in tqdm(train_list):
    shutil.copy(image_names[i], os.path.join(target_dir, 'train', 'image'))
    shutil.copy(mask_names[i], os.path.join(target_dir, 'train', 'mask'))
    shutil.copy(phase_names[i], os.path.join(target_dir, 'train', 'phase'))
print('Generating Test Set...')
for i in tqdm(test_list):
    shutil.copy(image_names[i], os.path.join(target_dir, 'test', 'image'))
    shutil.copy(mask_names[i], os.path.join(target_dir, 'test', 'mask'))
    shutil.copy(phase_names[i], os.path.join(target_dir, 'test', 'phase'))
