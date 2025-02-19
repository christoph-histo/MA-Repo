import os
import numpy as np
import torch
from torch.utils.data import Dataset
import json
import fnmatch
import Data_augmentation_3D as da3d

class VolumeToPatchesDataset(Dataset):
    def __init__(self, root_dir, transform=None, test = False, num_channels = 3, augmentaiton = None):
        self.root_dir = root_dir
        self.transform = transform
        self.augmentaiton = augmentaiton
        self.samples = []  
        self.count_samples_per_class = {0:0, 1:0, 2:0}
        self.num_channels = num_channels
        if test:
            self.class_to_idx = { 'lung_test': 0, 'skin_test': 1, 'intestine_test': 2 }
        else:
            self.class_to_idx = { 'lung': 0, 'skin': 1, 'intestine': 2 }
        # Populate self.samples with (slice_path, label) pairs
        for class_name, class_idx in self.class_to_idx.items():
                    class_dir = os.path.join(root_dir, class_name)
                    mask_dir = class_dir+"_tissue_segmentation"
                    print(class_dir)
                    count = 0
                    for volume_file in os.listdir(class_dir):
                        if volume_file.endswith('.raw'):
                            count += 1

                            volume_path = os.path.join(class_dir, volume_file)

                            mask_path = self.find_mask_file(mask_dir, volume_file.replace(".raw", ""))

                            shape, z_min, z_max = self.read_json(volume_path)
                            
                            volume = np.fromfile(volume_path, dtype= ">u2").reshape(shape,order = 'F')

                            mask_shape = (shape[0] // 4, shape[1] // 4, shape[2] // 4)
                            mask = np.fromfile(mask_path, dtype=">u2").reshape(mask_shape, order='F')
                            
                            # Append each 3D patch as a separate sample
                            for slice_index in range(z_min, z_max, 32):
                                volume_patch = volume[:, :, slice_index:slice_index + 32]
                                mask_patch = mask[:, :, (slice_index // 4):(slice_index // 4) + 8]
                                patches = self.extract_patches(volume_patch, mask_patch)
                                for patch in patches:
                                    self.samples.append((patch, class_idx))
                                    self.count_samples_per_class[class_idx] += 1

                            del volume


    def find_mask_file(self, directory, partial_name):

        for root, dirs, files in os.walk(directory):
            for file in files:
                if fnmatch.fnmatch(file, f"{partial_name}*"):
                    if file.endswith('segmentation_tissue.raw'):
                        return os.path.join(root, file)
        return None
    
    def extract_patches(self, volume, mask, patch_size=(128, 128, 32), threshold=0.01):
        patch_height, patch_width, patch_depth = patch_size
        # Check if the volume is roughly 4 times in each dimension the size of the mask
        assert volume.shape[0] // 4 == mask.shape[0] and volume.shape[1] // 4 == mask.shape[1] and volume.shape[2] // 4 == mask.shape[2]

        patches = []
        for i in range(0, volume.shape[0], patch_height):
            for j in range(0, volume.shape[1], patch_width):
                for k in range(0, volume.shape[2], patch_depth):
                    # Calculate the patch boundaries
                    start_x = i
                    end_x = min(start_x + patch_height, volume.shape[0])
                    start_y = j
                    end_y = min(start_y + patch_width, volume.shape[1])
                    start_z = k
                    end_z = min(start_z + patch_depth, volume.shape[2])

                    # Adjust the start position to ensure the patch size is maintained
                    if end_x - start_x < patch_height:
                        start_x = end_x - patch_height
                    if end_y - start_y < patch_width:
                        start_y = end_y - patch_width
                    if end_z - start_z < patch_depth:
                        start_z = end_z - patch_depth

                    # Extract the patch from the volume and mask
                    volume_patch = volume[start_y:end_y, start_x:end_x, start_z:end_z]
                    # Calculate the corresponding region in the mask
                    mask_patch = mask[start_y // 4:end_y // 4, start_x // 4:end_x // 4, start_z // 4:end_z // 4]

                    # Check if the percentage of ones in the mask patch meets the threshold
                    if np.mean(mask_patch) >= threshold:
                        patches.append(volume_patch)

        return patches


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patch, label = self.samples[idx]

        patch = patch.astype(np.float32)

        # Convert the 3D patch to a tensor
        patch_tensor = torch.tensor(patch, dtype=torch.float32)

        if self.num_channels == 3:
            patch_tensor = patch_tensor.unsqueeze(0).repeat(3, 1, 1, 1)
        elif self.num_channels == 1:
            patch_tensor = patch_tensor.unsqueeze(0)
        # Apply transformations if specified
        if self.transform:
            patch_tensor = self.transform(patch_tensor)
        
        # Convert the label to a tensor
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return patch_tensor, label_tensor
    

    def read_json(self,raw_volume_path):
        print(raw_volume_path)

        json_file_name = raw_volume_path.replace('.raw','.json')
        # Load the JSON data from a file
        with open(json_file_name, 'r') as f: 
            data = json.load(f)

        # Extract volume shape from the JSON data
        volume_info = data.get('volume', {})


        # Get the original shape values
        nx = volume_info.get('nx', 0)  
        ny = volume_info.get('ny', 0)
        nz = volume_info.get('nz', 0)
        z_min = volume_info.get('z_tissue_min')
        z_max = volume_info.get('z_tissue_max')
        shape = (nx, ny, nz)

        return shape, z_min, z_max
    
    
