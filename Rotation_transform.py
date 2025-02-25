import torch
import random

class RandomRotate3D(object):
    def __call__(self, volume):
        """
        volume: can be a 3D tensor (H, W, D) or a 4D tensor (C, H, W, D).
        
        We apply one of the four possible transforms with probability 1/4 each:
          1) Rotate 90° around z-axis
          2) Rotate 180° around z-axis
          3) Rotate 270° around z-axis
          4) Flip along z-axis
          
        Returns:
            volume (Tensor): The transformed tensor, same shape as input.
        """
        # Decide which type of transform to apply
        choice = random.randint(0, 3)
        
        # Determine if volume has a channel dimension
        if volume.ndim == 3:
            # volume shape: (H, W, D)
            # (H, W) = (0, 1), D = 2
            rot_dims = (0, 1)
            flip_dim = (2,)
        elif volume.ndim == 4:
            # volume shape: (C, H, W, D)
            # (H, W) = (1, 2), D = 3
            rot_dims = (1, 2)
            flip_dim = (3,)
        else:
            raise ValueError(
                f"Expected 3D (H, W, D) or 4D (C, H, W, D) input, got shape {volume.shape}"
            )
        
        if choice == 0:
            # Rotate 90° around z-axis
            volume = torch.rot90(volume, k=1, dims=rot_dims)
        elif choice == 1:
            # Rotate 180° around z-axis
            volume = torch.rot90(volume, k=2, dims=rot_dims)
        elif choice == 2:
            # Rotate 270° around z-axis
            volume = torch.rot90(volume, k=3, dims=rot_dims)
        else:
            # Flip along z-axis
            volume = torch.flip(volume, dims=flip_dim)
        
        return volume

class RandomRotate2D(object):
    def __call__(self, patch):

        k = random.randint(1, 3)

        patch = torch.rot90(patch, k=k, dims=(2, 3))

        return patch
