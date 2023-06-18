import h5py, torch, utils
import numpy as np
# from PIL import Image

# def rotate_image(image):
#     return image.rotate(-90, expand=True)

class NYU_Depth_V2_Labeled_Dataset(torch.utils.data.Dataset):
    """Python interface for the labeled subset of the NYU dataset.
    
    Return the color and depth image in the form of numpy array: color array (3,480,640), depth array (1,480,640)

    To save memory, call the `close()` method of this class to close
    the dataset file once you're done using it.
    """

    def __init__(self, path, virtual_depth_planes, return_type='image_mask_id'):
        """Opens the labeled dataset file at the given path."""
        self.file = h5py.File(path, mode='r')
        self.color_maps = self.file['images']
        self.depth_maps = self.file['depths']
        self.virtual_depth_planes = virtual_depth_planes
        self.return_type = return_type

    def close(self):
        """Closes the HDF5 file from which the dataset is read."""
        self.file.close()

    def __len__(self):
        return len(self.color_maps)

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise StopIteration
        color_array = self.color_maps[idx]
        color_array = np.moveaxis(color_array, 0, -1)
        color_array = np.transpose(color_array, (2,1,0))
        color_array = color_array[1,:,:]
        color_tensor = torch.from_numpy(color_array)
        # color_image = Image.fromarray(color_map, mode='RGB')
        # color_image = rotate_image(color_image)

        depth_array = self.depth_maps[idx]
        depth_array = np.transpose(depth_array)
        depth_array = depth_array[np.newaxis, ...]
        depth_tensor = torch.from_numpy(depth_array)
        # depth_image = Image.fromarray(depth_map, mode='F')
        # depth_image = rotate_image(depth_image)
        
        mask = utils.decompose_depthmap(depth_tensor, self.virtual_depth_planes)

        if self.return_type == 'image_mask_id':
            return color_tensor, mask, f'NYU_labeled_{idx}'
        elif self.return_type == 'image_depth_id':
            return color_tensor, depth_tensor, f'NYU_labeled_{idx}'


if __name__ == '__main__':
    nyu_dataset = NYU_Depth_V2_Labeled_Dataset('/media/datadrive/NYU_Depth_V2/nyu_depth_v2_labeled.mat', [1, 3.5, 5.5])
    image, mask, id = nyu_dataset[170]
    pass