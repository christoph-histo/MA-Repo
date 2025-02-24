import os
import numpy as np
import torch
import torchvision.transforms as transform
from torch.utils.data import Dataset
import torchio.transforms as tio
import json
from PIL import Image
import  Rotation_transform

class VolumeToSliceDataset(Dataset):
    def __init__(self, root_dir, transform=None, test = False, augmentation = None):
        self.root_dir = root_dir
        self.transform = transform
        self.augmentation = augmentation
        self.samples = []  
        self.count_samples_per_class = {0:0, 1:0, 2:0}
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
                            for slice_index in range(z_min, z_max):
                                self.samples.append((volume[:,:,slice_index], class_idx))
                                self.count_samples_per_class[class_idx] += 1

                            #sanity check
                            #for slice_index in range(shape[2]):
                            #    if slice_index < z_min or slice_index > z_max:
                            #        self.samples.append((volume[:,:,slice_index], class_idx))

                            del volume


    def __len__(self):
        if self.augmentation:
            return len(self.samples) * 2
        else:
            return len(self.samples)


    def __getitem__(self, idx):

        aug = False
        if len(self.samples) > idx:
            slice_array, label = self.samples[idx]
        else:
            slice_array, label = self.samples[idx % len(self.samples)]
            aug = True

        # Convert the slice to a PIL Image (assuming grayscale)

        if slice_array.dtype != np.uint8:
            slice_array = (slice_array / slice_array.max() * 255).astype(np.uint8)
        
        slice_image = Image.fromarray(slice_array)
        
        # Convert to RGB for compatibility with models expecting 3 channels
        slice_image = slice_image.convert("RGB")

        slice_image = transform.PILToTensor()(slice_image)
        # Normalize the slice_array
        slice_array = (slice_array - slice_array.min()) / (slice_array.max() - slice_array.min())

        # Apply transformations if specified
        if self.transform:
            slice_image = self.transform(slice_image)

        if aug:
            if self.augmentation == 'elastic':
                slice_image = slice_image.unsqueeze(0)
                slice_image = tio.RandomElasticDeformation(num_control_points=(5,30,30),locked_borders=2,max_displacement=(0,30,30))(slice_image)
                slice_image = slice_image.squeeze(0)
            elif self.augmentaiton == 'tripath':
                transforms = {
                    Rotation_transform.RandomRotate2D() : 2/3, 
                    tio.Gamma(gamma=(0.8,1.2)) : 1/3
                }
                slice_image = slice_image.unsqueeze(0)
                slice_image = tio.OneOf(transforms)(slice_image)
                slice_image = slice_image.squeeze(0)
            else:
                print("Error: Augmentation not supported")
                return

        
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


