import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import json
from PIL import Image

class VolumeToSliceDataset(Dataset):
    def __init__(self, root_dir, transform=None, test = False):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  
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

                            mask_path = self.find_mask_file(mask_dir, volume_file)

                            shape, z_min, z_max = self.read_json(volume_path)
                            
                            volume = np.fromfile(volume_path, dtype= ">u2").reshape(shape,order = 'F')

                            mask  = np.fromfile(mask_path, dtype= ">u2").reshape(shape//4,order = 'F')
                            
                            # Append each slice index as a separate sample
                            for slice_index in range(z_min, z_max):
                                slice_array = volume[:, :, slice_index]
                                mask_array = mask[:, :, slice_index // 4]
                                patches = self.extract_patches(slice_array, mask_array)
                                for patch in patches:
                                    self.samples.append((patch, class_idx))

                            del volume

                    
    def find_mask_file(self, directory, partial_name):

        for root, dirs, files in os.walk(directory):
            for file in files:
                if fnmatch.fnmatch(file, f"{partial_name}*"):
                    return os.path.join(root, file)
        return None
    

    def extract_patches(self, image, mask, patch_size=(128, 128), threshold=0.1):
        patch_height, patch_width = patch_size
        #check if the image is rougly 4 times in each dimension the size of the mask
        assert image.shape[0] // 4 == mask.shape[0] and image.shape[1] // 4 == mask.shape[1]

        patches = []
        for i in range(0, image.shape[0], patch_height):
            for j in range(0, image.shape[1], patch_width):
                # Calculate the patch boundaries
                start_x = i
                end_x = min(start_x + patch_height, image.shape[0])
                start_y = j
                end_y = min(start_y + patch_width, image.shape[1])

                # Adjust the start position to ensure the patch size is maintained
                if end_x - start_x < patch_height:
                    start_x = end_x - patch_height
                if end_y - start_y < patch_width:
                    start_y = end_y - patch_width

                # Extract the patch from the image and mask
                image_patch = image[start_y:end_y, start_x:end_x]
                # Calculate the corresponding region in the mask
                mask_patch = mask[start_y // 4:end_y // 4, start_x // 4:end_x // 4]

                # Check if the percentage of ones in the mask patch meets the threshold
                if np.mean(mask_patch) >= threshold:
                    patches.append(image_patch)

        return patches



    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        slice_array, label = self.samples[idx]

        # Convert the slice to a PIL Image (assuming grayscale)
        slice_image = Image.fromarray(slice_array)
        
        # Convert to RGB for compatibility with models expecting 3 channels
        slice_image = slice_image.convert("RGB")

        # Apply transformations if specified
        if self.transform:
            slice_image = self.transform(slice_image)
        
        # Convert the label to a tensor
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return slice_image, label_tensor
    

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
