import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
import json
from PIL import Image

class VolumeToSliceDataset(Dataset):
    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  
        self.class_to_idx = { 'lung': 0, 'skin': 1, 'intestine': 2 }
        
        # Populate self.samples with (slice_path, label) pairs
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = os.path.join(root_dir, class_name)
            for volume_file in os.listdir(class_dir):
                if volume_file.endswith('.raw'):
                    volume_path = os.path.join(class_dir, volume_file)
                    self.samples.append((volume_path, class_idx))


    def __len__(self):
        return len(self.samples)

    def load_volume(self, file_path):
        shape = self.read_json(file_path)
        with open(file_path, 'rb') as f:
            volume = np.fromfile(f, dtype= ">u2").reshape(shape,order = 'F')
        return volume

    def __getitem__(self, idx):
        volume_path, label = self.samples[idx]
        
        # Load 3D volume
        volume = self.load_volume(volume_path)
        
        # Select a random slice index along the z-axis
        random_slice_index = random.randint(0, volume.shape[2] - 1)
        slice_ = volume[:, :, random_slice_index]
        
        # Convert the selected slice to a PIL Image (assumes grayscale)
        slice_ = Image.fromarray(slice_)
        slice_ = slice_.convert("RGB")  # Convert to 3 channels for ResNet compatibility

        # Apply the transformation if specified
        if self.transform:
            slice_ = self.transform(slice_)
        
        # Convert the label to tensor
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return slice_, label_tensor
    
    def read_json(self,raw_volume_path):

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

        shape = (nx, ny, nz)

        return shape


