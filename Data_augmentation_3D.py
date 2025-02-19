import math
import torch
import torch.nn.functional as F



def elastic_transform(tensor, alpha, sigma, mode='bilinear', padding_mode='reflection', align_corners=True):

    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)  # now (C, H, W, D) with C==1

    C, H, W, D = tensor.shape

    # grid_sample for 3D expects (N, C, D, H, W), so add batch and permute:
    # From (C, H, W, D) --> (1, C, D, H, W)
    tensor = tensor.unsqueeze(0).permute(0, 1, 4, 2, 3)
    N, C, D, H, W = tensor.shape  # update dimensions

    # --- 1. Generate the displacement field ---
    # Create random noise in [-1, 1] for each spatial axis (x, y, z)
    # Shape: (1, 3, D, H, W)
    disp = torch.rand(1, 3, D, H, W, device=tensor.device) * 2 - 1

    # Create a 3D Gaussian kernel.
    kernel_size = int(2 * math.ceil(3 * sigma) + 1)
    # Create a 1D kernel
    ax = torch.arange(kernel_size, dtype=torch.float32, device=tensor.device) - (kernel_size - 1) / 2.
    gauss = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()

    # Create 3D kernel by outer product
    kernel3d = gauss[:, None, None] * gauss[None, :, None] * gauss[None, None, :]
    # Reshape to (1, 1, k, k, k) and then repeat for 3 channels (using groups in conv3d)
    kernel3d = kernel3d[None, None, :, :, :].repeat(3, 1, 1, 1, 1)

    # Smooth the displacement field with the Gaussian kernel using groups=3 so that each channel is convolved separately.
    disp = F.conv3d(disp, kernel3d, padding=kernel_size // 2, groups=3) * alpha
    # Rearrange to shape (1, D, H, W, 3) so that the last dimension gives (dx, dy, dz)
    disp = disp.permute(0, 2, 3, 4, 1)

    # --- 2. Create the base grid ---
    # We create a meshgrid for the coordinates.
    d_lin = torch.linspace(0, D - 1, D, device=tensor.device)
    h_lin = torch.linspace(0, H - 1, H, device=tensor.device)
    w_lin = torch.linspace(0, W - 1, W, device=tensor.device)
    grid_d, grid_h, grid_w = torch.meshgrid(d_lin, h_lin, w_lin, indexing='ij')
    # grid_sample expects grid[..., 0] = x (width), grid[..., 1] = y (height), grid[..., 2] = z (depth)
    base_grid = torch.stack((grid_w, grid_h, grid_d), dim=-1)  # shape: (D, H, W, 3)
    base_grid = base_grid.unsqueeze(0)  # add batch dim -> (1, D, H, W, 3)

    # Add the displacement to the base grid
    new_grid = base_grid + disp

    # --- 3. Normalize the grid ---
    # Normalize coordinates to the range [-1, 1]
    new_grid[..., 0] = 2.0 * new_grid[..., 0] / (W - 1) - 1.0  # x: width
    new_grid[..., 1] = 2.0 * new_grid[..., 1] / (H - 1) - 1.0  # y: height
    new_grid[..., 2] = 2.0 * new_grid[..., 2] / (D - 1) - 1.0  # z: depth

    # --- 4. Sample the image using grid_sample ---
    out = F.grid_sample(tensor, new_grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)

    # --- 5. Convert back to original shape (C, H, W, D) ---
    # Currently out has shape (1, C, D, H, W)
    out = out.squeeze(0).permute(0, 2, 3, 1)  # (C, H, W, D)
    return out

import torch
import torch.nn.functional as F

def elastic_transform_3d(tensor, alpha=1, sigma=0.2, seed=None):
    
    if seed is not None:
        torch.manual_seed(seed)

    C, H, W, D = tensor.shape

    # Generate random displacement fields
    displacement = torch.randn(3, H, W, D)  # (dx, dy, dz) for each point

    # Apply Gaussian smoothing to displacements
    kernel_size = int(2 * sigma + 0.5)
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure kernel size is odd

    # Create a Gaussian kernel
    kernel = torch.exp(-torch.linspace(-1, 1, kernel_size) ** 2 / (2 * sigma ** 2))
    kernel /= kernel.sum()  # Normalize

    # Apply 3D Gaussian filtering
    for i in range(3):  # Apply to dx, dy, dz
        displacement[i] = F.conv3d(
            displacement[i].unsqueeze(0).unsqueeze(0),  # Shape (1,1,H,W,D)
            kernel.view(1, 1, kernel_size, 1, 1),  # Apply along H
            padding=(kernel_size // 2, 0, 0)
        ).squeeze(0).squeeze(0)
        displacement[i] = F.conv3d(
            displacement[i].unsqueeze(0).unsqueeze(0),
            kernel.view(1, 1, 1, kernel_size, 1),  # Apply along W
            padding=(0, kernel_size // 2, 0)
        ).squeeze(0).squeeze(0)
        displacement[i] = F.conv3d(
            displacement[i].unsqueeze(0).unsqueeze(0),
            kernel.view(1, 1, 1, 1, kernel_size),  # Apply along D
            padding=(0, 0, kernel_size // 2)
        ).squeeze(0).squeeze(0)

    # Normalize displacement field
    displacement = displacement / displacement.abs().max()

    # Scale displacement field by alpha
    displacement *= alpha

    # Generate coordinate grid
    grid_x, grid_y, grid_z = torch.meshgrid(
        torch.linspace(-1, 1, H),
        torch.linspace(-1, 1, W),
        torch.linspace(-1, 1, D),
        indexing="ij"
    )

    # Stack and apply displacement
    grid = torch.stack((grid_x, grid_y, grid_z))  # Shape (3, H, W, D)
    grid += displacement / torch.tensor([H, W, D]).view(3, 1, 1, 1) * 2  # Normalize displacement

    # Reshape grid for grid_sample (shape: (1, H, W, D, 3))
    grid = grid.permute(1, 2, 3, 0).unsqueeze(0)

    # Apply transformation using grid_sample
    transformed = F.grid_sample(
        tensor.unsqueeze(0),  # Shape (1, C, H, W, D)
        grid,
        mode="bilinear",
        padding_mode="reflection",
        align_corners=True
    ).squeeze(0)

    return transformed


def rotate(image, angle):
    augmented_image = transforms.Rotate(angle)(image)
    return augmented_image
