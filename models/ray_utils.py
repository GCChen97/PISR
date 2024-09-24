import torch
import numpy as np

from kornia.geometry.calibration.distort import tilt_projection, distort_points
from kornia.geometry.linalg import transform_points



# from Kornia
def get_undistorted_ray_directions(H: int, W: int, K: torch.Tensor, dist: torch.Tensor) \
    -> torch.Tensor:
    """From  distorted image points to camera rays

    Args:
        H (int): image height
        W (int): image width
        K (torch.Tensor): camera matrix, [3,3]
        dist (torch.Tensor): distortion coefficients [n]

    Returns:
        rays (torch.Tensor): [N, 3]
    """
    K = K[None]
    dist = dist[None]
    use_pixel_centers = True
    pixel_center = 0.5 if use_pixel_centers else 0
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32) + pixel_center,
        np.arange(H, dtype=np.float32) + pixel_center,
        indexing='xy'
    )
    u, v = torch.from_numpy(i), torch.from_numpy(j)
    u = u[None]
    v = v[None]

    num_iters = 5
    # Adding zeros to obtain vector with 14 coeffs.
    if dist.shape[-1] < 14:
        dist = torch.nn.functional.pad(dist, [0, 14 - dist.shape[-1]])

    # Convert 2D points from pixels to normalized camera coordinates
    cx: torch.Tensor = K[..., 0:1, 2]  # princial point in x (Bx1)
    cy: torch.Tensor = K[..., 1:2, 2]  # princial point in y (Bx1)
    fx: torch.Tensor = K[..., 0:1, 0]  # focal in x (Bx1)
    fy: torch.Tensor = K[..., 1:2, 1]  # focal in y (Bx1)

    # This is equivalent to K^-1 [u,v,1]^T
    x: torch.Tensor = (u - cx) / fx  # (BxN - Bx1)/Bx1 -> BxN
    y: torch.Tensor = (v - cy) / fy  # (BxN - Bx1)/Bx1 -> BxN

    # Compensate for tilt distortion
    if torch.any(dist[..., 12] != 0) or torch.any(dist[..., 13] != 0):
        inv_tilt = tilt_projection(dist[..., 12], dist[..., 13], True)

        # Transposed untilt points (instead of [x,y,1]^T, we obtain [x,y,1])
        x, y = transform_points(inv_tilt, torch.stack([x, y], dim=-1)).unbind(-1)

    # Iteratively undistort points
    x0, y0 = x, y
    for _ in range(num_iters):
        r2 = x * x + y * y
        inv_rad_poly = (1 + dist[..., 5:6] * r2 + dist[..., 6:7] * r2 * r2 + dist[..., 7:8] * r2**3) / (
            1 + dist[..., 0:1] * r2 + dist[..., 1:2] * r2 * r2 + dist[..., 4:5] * r2**3
        )
        deltaX = (
            2 * dist[..., 2:3] * x * y
            + dist[..., 3:4] * (r2 + 2 * x * x)
            + dist[..., 8:9] * r2
            + dist[..., 9:10] * r2 * r2
        )
        deltaY = (
            dist[..., 2:3] * (r2 + 2 * y * y)
            + 2 * dist[..., 3:4] * x * y
            + dist[..., 10:11] * r2
            + dist[..., 11:12] * r2 * r2
        )

        x = (x0 - deltaX) * inv_rad_poly
        y = (y0 - deltaY) * inv_rad_poly

    return torch.stack([x, -y, -torch.ones_like(u)], -1).squeeze(0)
# end copy

def cast_rays(ori, dir, z_vals):
    return ori[..., None, :] + z_vals[..., None] * dir[..., None, :]


def get_ray_directions(W, H, fx, fy, cx, cy, use_pixel_centers=True):
    pixel_center = 0.5 if use_pixel_centers else 0
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32) + pixel_center,
        np.arange(H, dtype=np.float32) + pixel_center,
        indexing='xy'
    )
    i, j = torch.from_numpy(i), torch.from_numpy(j)

    directions = torch.stack([(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1) # (H, W, 3)

    return directions


def get_rays(directions, c2w, keepdim=False):
    # Rotate ray directions from camera coordinate to the world coordinate
    # rays_d = directions @ c2w[:, :3].T # (H, W, 3) # slow?
    assert directions.shape[-1] == 3

    if directions.ndim == 2: # (N_rays, 3)
        assert c2w.ndim == 3 # (N_rays, 4, 4) / (1, 4, 4)
        rays_d = (directions[:,None,:] * c2w[:,:3,:3]).sum(-1) # (N_rays, 3)
        rays_o = c2w[:,:,3].expand(rays_d.shape)
    elif directions.ndim == 3: # (H, W, 3)
        if c2w.ndim == 2: # (4, 4)
            rays_d = (directions[:,:,None,:] * c2w[None,None,:3,:3]).sum(-1) # (H, W, 3)
            rays_o = c2w[None,None,:,3].expand(rays_d.shape)
        elif c2w.ndim == 3: # (B, 4, 4)
            rays_d = (directions[None,:,:,None,:] * c2w[:,None,None,:3,:3]).sum(-1) # (B, H, W, 3)
            rays_o = c2w[:,None,None,:,3].expand(rays_d.shape)

    if not keepdim:
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    return rays_o, rays_d
