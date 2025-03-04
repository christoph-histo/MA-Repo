import os
import numpy as np
import torch
from torch.utils.data import Dataset
import json
import fnmatch
import torchio.transforms as tio    
import Rotation_transform
import random

class VolumeToFeaturesDataset(Dataset):
    def __init__(self, root_dir, transform=None, test=False, num_channels=3, augmentation=None, encoder=None, SwinUnetr=False):
        self.root_dir = root_dir
        self.transform = transform
        self.augmentation = augmentation
        self.max_idx_per_volume = {}
        
        self.samples_tree = {}

        self.label_per_volume = {}
        
        self.t = {}
    
        self.encoder = encoder
        self.num_channels = num_channels
        self.SwinUnetr = SwinUnetr

        if test:
            self.class_to_idx = {'lung_test': 0, 'skin_test': 1, 'intestine_test': 2}
        else:
            self.class_to_idx = {'lung': 0, 'skin': 1, 'intestine': 2}

        cumulative_idx = 0

        for class_name, class_idx in self.class_to_idx.items():
            class_dir = os.path.join(root_dir, class_name)
            mask_dir = class_dir + "_tissue_segmentation"

            print("Processing directory:", class_dir)
            if not os.path.isdir(class_dir):
                print(f"[WARNING] {class_dir} does not exist, skipping.")
                continue

            for volume_file in os.listdir(class_dir):
                if volume_file.endswith('.raw'):
                    volume_path = os.path.join(class_dir, volume_file)
                    mask_path = self.find_mask_file(mask_dir, volume_file.replace(".raw", ""))

                    shape, z_min, z_max = self.read_json(volume_path)

                    volume = np.fromfile(volume_path, dtype=">u2").reshape(shape, order='F')
                    mask_shape = (shape[0] // 4, shape[1] // 4, shape[2] // 4)
                    mask = np.fromfile(mask_path, dtype=">u2").reshape(mask_shape, order='F')

                    # Initialize the list of patches for this volume
                    self.samples_tree[volume_file] = []

                    # Extract patches for this volume
                    for slice_index in range(z_min, z_max, 32):

                        volume_patch = volume[:, :, slice_index : slice_index + 32]

                        mask_patch = mask[:, :, (slice_index // 4) : (slice_index // 4) + 8]

                        patches = self.extract_patches(volume_patch, mask_patch)
                        self.samples_tree[volume_file].extend(patches)
                    
                    random.shuffle(self.samples_tree[volume_file])

                    self.label_per_volume[volume_file] = class_idx

                    num_patches = len(self.samples_tree[volume_file])

                    num_units = int(np.ceil(num_patches / 10))

                    cumulative_idx += num_units
                    self.max_idx_per_volume[volume_file] = cumulative_idx

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

        assert volume.shape[0] // 4 == mask.shape[0] and volume.shape[1] // 4 == mask.shape[1] and volume.shape[2] // 4 == mask.shape[2]

        patches = []
        for i in range(0, volume.shape[0], patch_height):
            for j in range(0, volume.shape[1], patch_width):
                for k in range(0, volume.shape[2], patch_depth):
                    start_x = i
                    end_x = min(start_x + patch_height, volume.shape[0])
                    start_y = j
                    end_y = min(start_y + patch_width, volume.shape[1])
                    start_z = k
                    end_z = min(start_z + patch_depth, volume.shape[2])

                    if end_x - start_x < patch_height:
                        start_x = end_x - patch_height
                    if end_y - start_y < patch_width:
                        start_y = end_y - patch_width
                    if end_z - start_z < patch_depth:
                        start_z = end_z - patch_depth

                    volume_patch = volume[start_y:end_y, start_x:end_x, start_z:end_z]
                    mask_patch = mask[start_y // 4:end_y // 4, start_x // 4:end_x // 4, start_z // 4:end_z // 4]

                    if np.mean(mask_patch) >= threshold:
                        patches.append(volume_patch)

        return patches


    def __len__(self):
        if not self.max_idx_per_volume:
            return 0
        
        max_idx = max(self.max_idx_per_volume.values())
        
        if self.augmentation:
            return 2 * max_idx
        else:
            return max_idx
    
    def rotate_function(self,tensor):
        return Rotation_transform.RandomRotate3D()(tensor)
    
    def create_feats(self, tensor):
        with torch.no_grad():
            feature = self.encoder(tensor.unsqueeze(0))
        return feature.squeeze(0)
    
    def __getitem__(self, idx):

        max_raw_idx = max(self.max_idx_per_volume.values()) if self.max_idx_per_volume else 0

        if idx >= max_raw_idx:
            aug = True
            idx = idx - max_raw_idx
        else:
            aug = False

        volume_maxes = sorted(self.max_idx_per_volume.items(), key=lambda x: x[1])

        prev_cumulative = 0
        chosen_volume = None

        for vol_file, cum_val in volume_maxes:
            if idx < cum_val:
                chosen_volume = vol_file
                break
            prev_cumulative = cum_val


        if chosen_volume is None:

            raise IndexError(f"Index {idx} out of range in dataset")

        local_index = idx - prev_cumulative
        patch_start = local_index * 10
        patch_end = patch_start + 10

        patches_for_volume = self.samples_tree[chosen_volume]  
        label_for_volume = self.label_per_volume[chosen_volume]

        total_patches_in_vol = len(patches_for_volume)

        if patch_start >= total_patches_in_vol:
            patch_start = patch_start - 10
            patch_end   = patch_start + 10

        chosen_patches = []
        
        while len(chosen_patches) < 10:
            current_end = min(patch_end, total_patches_in_vol)
            chosen_patches.extend(patches_for_volume[patch_start:current_end])

            needed_more = 10 - len(chosen_patches)
            if needed_more > 0:
                patch_start = 0
                patch_end = needed_more
            else:
                break

        chosen_patches = chosen_patches[:10]  

        feature_list = []
        for patch_data in chosen_patches:
            patch_data = patch_data.astype(np.float32)
            patch_tensor = torch.tensor(patch_data)

            if self.transform:
                patch_tensor = self.transform(patch_tensor)

            if aug:
                if self.augmentation == 'elastic':
                    patch_tensor = patch_tensor.unsqueeze(0) 
                    patch_tensor = tio.RandomElasticDeformation(
                        num_control_points=(9, 9, 5),
                        max_displacement=(10, 10, 2)
                    )(patch_tensor)
                    patch_tensor = patch_tensor.squeeze(0)

                elif self.augmentation == 'tripath':
                    transforms = {
                        tio.Lambda(self.rotate_function): 2/3,
                        tio.Gamma(gamma=(0.8,1.2)): 1/3
                    }
                    patch_tensor = patch_tensor.unsqueeze(0)
                    patch_tensor = tio.OneOf(transforms)(patch_tensor)
                    patch_tensor = patch_tensor.squeeze(0)
                else:
                    print("Error: Augmentation not supported")

            if self.num_channels == 3:
                patch_tensor = patch_tensor.unsqueeze(0).repeat(3, 1, 1, 1)
            elif self.num_channels == 1:
                if self.SwinUnetr:
                    patch_tensor = patch_tensor.unsqueeze(0)
                else:
                    patch_tensor = patch_tensor.unsqueeze(0)

            feature = self.create_feats(patch_tensor)

            feature_list.append(feature)

        aggregator_input = torch.stack(feature_list, dim=0)


        label_tensor = torch.tensor(label_for_volume, dtype=torch.long)

        return aggregator_input, label_tensor

    

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
    
    
