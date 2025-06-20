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
                    print(class_dir)
                    count = 0
                    for volume_file in os.listdir(class_dir):
                        if volume_file.endswith('.raw'):
                            count += 1

                            volume_path = os.path.join(class_dir, volume_file)

                            shape, z_min, z_max = self.read_json(volume_path)
                            
                            volume = np.fromfile(volume_path, dtype= ">u2").reshape(shape,order = 'F')
                            
                            # Append each slice index as a separate sample
                            for slice_index in range(shape[2]):
                                self.samples.append((volume[:,:,slice_index], class_idx))

                            del volume


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        slice_array, label = self.samples[idx]

        # Convert the slice to a PIL Image (assuming grayscale)

        if slice_array.dtype != np.uint8:
            slice_array = (slice_array / slice_array.max() * 255).astype(np.uint8)
        
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


