#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)
    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def getProjectionMatrixK(K, H, W, znear, zfar):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    s = K[0, 1]

    P = torch.zeros((4, 4))

    z_sign = 1.0

    P[0, 0] = 2 * fx / W
    P[0, 1] = 2 * s / W
    P[0, 2] = -1 + 2 * (cx / W)

    P[1, 1] = 2 * fy / H
    P[1, 2] = -1 + 2 * (cy / H)

    P[2, 2] = z_sign * (zfar + znear) / (zfar - znear)
    P[2, 3] = -1 * z_sign * 2 * zfar * znear / (zfar - znear)
    P[3, 2] = z_sign

    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))

def project_numpy(xyz, K, RT, H, W):
    '''
    input: 
    xyz: [N, 3], pointcloud
    K: [3, 3], intrinsic
    RT: [4, 4], w2c
    
    output:
    mask: [N], pointcloud in camera frustum
    xy: [N, 2], coord in image plane
    '''
    
    xyz_cam = np.dot(xyz, RT[:3, :3].T) + RT[:3, 3:].T
    valid_depth = xyz_cam[:, 2] > 0
    xyz_pixel = np.dot(xyz_cam, K.T)
    xyz_pixel = xyz_pixel[:, :2] / xyz_pixel[:, 2:]
    valid_x = np.logical_and(xyz_pixel[:, 0] >= 0, xyz_pixel[:, 0] < W)
    valid_y = np.logical_and(xyz_pixel[:, 1] >= 0, xyz_pixel[:, 1] < H)
    valid_pixel = np.logical_and(valid_x, valid_y)
    mask = np.logical_and(valid_depth, valid_pixel)
    
    return xyz_pixel, mask
    
def project_torch(xyz, K, RT, H, W):
    '''
    input: 
    xyz: [N, 3], pointcloud
    K: [3, 3], intrinsic
    RT: [4, 4], w2c
    
    output:
    mask: [N], pointcloud in camera frustum
    xy: [N, 2], coord in image plane
    '''
    
    xyz_cam = torch.matmul(xyz, RT[:3, :3].T) + RT[:3, 3:].T
    valid_depth = xyz_cam[:, 2] > 0
    xyz_pixel = torch.matmul(xyz_cam, K.T)
    xyz_pixel = xyz_pixel[:, :2] / xyz_pixel[:, 2:]
    valid_x = torch.logical_and(xyz_pixel[:, 0] >= 0, xyz_pixel[:, 0] < W)
    valid_y = torch.logical_and(xyz_pixel[:, 1] >= 0, xyz_pixel[:, 1] < H)
    valid_pixel = torch.logical_and(valid_x, valid_y)
    mask = torch.logical_and(valid_depth, valid_pixel)
    
    return xyz_pixel, mask

def sphere_intersection(rays_o, rays_d, center, radius):
    rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)
    b = np.sum((rays_o - center) * rays_d, axis=-1, keepdims=True)
    c = np.sum((rays_o - center) * (rays_o - center), axis=-1, keepdims=True) - radius ** 2
    
    nears = (-b - np.sqrt(b ** 2 - c))
    fars = (-b + np.sqrt(b ** 2 - c))
    
    nears = np.nan_to_num(nears, nan=0.0)
    fars = np.nan_to_num(fars, nan=1e3)
    
    p_sphere = rays_o + fars * rays_d 
    
    return p_sphere 

def get_rays(H, W, K, R, T, perturb=False):
    # calculate the camera origin
    rays_o = -np.dot(R.T, T).ravel()
    # calculate the world coodinates of pixels
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    
    if perturb:
        perturb_i = np.random.rand(H, W)
        perturb_j = np.random.rand(H, W)
        xy1 = np.stack([i + perturb_i, j + perturb_j, np.ones_like(i)], axis=2)
    else:
        xy1 = np.stack([i + 0.5, j + 0.5, np.ones_like(i)], axis=2)
    
    pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
    pixel_world = np.dot(pixel_camera - T.ravel(), R)
    # calculate the ray direction
    rays_d = pixel_world - rays_o[None, None]
    rays_d = rays_d / np.linalg.norm(rays_d, axis=2, keepdims=True)
    rays_o = np.broadcast_to(rays_o, rays_d.shape)
    return rays_o, rays_d

def get_rays_torch(H, W, K, R, T, perturb=False):
    # calculate the camera origin
    rays_o = -torch.matmul(R.T, T).squeeze()
    # calculate the world coodinates of pixels
    i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32, device=K.device),
                       torch.arange(H, dtype=torch.float32, device=K.device),
                       indexing='xy')

    if perturb:
        perturb_i = torch.rand(H, W, device=K.device)
        perturb_j = torch.rand(H, W, device=K.device)
        xy1 = torch.stack([i + perturb_i, j + perturb_j, torch.ones_like(i)], dim=2)
    else:
        xy1 = torch.stack([i + 0.5, j + 0.5, torch.ones_like(i)], dim=2)
    
    pixel_camera = torch.matmul(xy1, torch.inverse(K).T)
    pixel_world = torch.matmul(pixel_camera - T.squeeze(), R)
    # calculate the ray direction
    rays_d = pixel_world - rays_o.unsqueeze(0).unsqueeze(0)
    rays_d = rays_d / torch.norm(rays_d, dim=2, keepdim=True)
    rays_o = rays_o.expand_as(rays_d)
    return rays_o, rays_d

