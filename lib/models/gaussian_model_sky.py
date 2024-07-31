import torch
import numpy as np
import os
from lib.config import cfg
from lib.utils.graphics_utils import BasicPointCloud
from lib.datasets.base_readers import fetchPly
from lib.models.gaussian_model import GaussianModel
from lib.utils.general_utils import quaternion_to_matrix

class GaussinaModelSky(GaussianModel):
    def __init__(
        self, 
        model_name='sky', 
        num_classes=1, 
        sphere_center=np.array([0, 0, 0]),
        sphere_radius=20,
    ):
        super().__init__(model_name=model_name, num_classes=num_classes)
        self.sphere_center = torch.from_numpy(sphere_center).float().cuda()
        self.sphere_radius = torch.Tensor([sphere_radius]).float().cuda()
        
    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):    
        print('Create sky model')
            
        pointcloud_path_sky = os.path.join(cfg.model_path, 'input_ply', 'points3D_sky.ply')
        assert os.path.exists(pointcloud_path_sky), f'Pointcloud {pointcloud_path_sky} does not exist'
        
        pcd_sky = fetchPly(pointcloud_path_sky)
        pointcloud_xyz = pcd_sky.points
        pointcloud_rgb = pcd_sky.colors
        pointcloud_normal = np.zeros_like(pointcloud_xyz)
        pcd = BasicPointCloud(pointcloud_xyz, pointcloud_rgb, pointcloud_normal)
            
        return super().create_from_pcd(pcd, self.sphere_radius.item())
    
    def get_extent(self):
        max_scaling = torch.max(self.get_scaling, dim=1).values

        extent_lower_bound = torch.topk(max_scaling, int(self.get_xyz.shape[0] * 0.1), largest=False).values[-1] / self.percent_dense
        extent_upper_bound = torch.topk(max_scaling, int(self.get_xyz.shape[0] * 0.1), largest=True).values[-1] / self.percent_dense
        
        extent = torch.clamp(self.sphere_radius, min=extent_lower_bound, max=extent_upper_bound)        
        print(f'extent: {extent.item()}, extent bound: [{extent_lower_bound}, {extent_upper_bound}]')

        # distance = torch.linalg.norm(self.get_xyz - self.scene_center, dim=1)
        # scale_factor = torch.clamp(distance / self.scene_radius - 1, min=1., max=3.)
        # extent = self.scene_radius * scale_factor
        return extent
    
    @property
    def get_scaling(self):
        scaling = self.scaling_activation(self._scaling)
        scaling = torch.clamp(scaling, max=self.sphere_radius.item())
        return scaling
    
    @property
    def get_xyz(self):
        xyz = self._xyz
        dists = torch.linalg.norm(xyz - self.sphere_center, dim=1, keepdims=True)
        ratios = dists / (2 * self.sphere_radius)
        xyz = torch.where(ratios < 1., self.sphere_center + (xyz - self.sphere_center) / ratios, xyz)
        return xyz
    
    def densify_and_prune(self, max_grad, min_opacity, prune_big_points):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.scalar_dict.clear()
        self.tensor_dict.clear()    
        self.scalar_dict['points_total'] = self.get_xyz.shape[0]
        print('=' * 20)
        print(f'Model name: {self.model_name}')
        print(f'Number of 3d gaussians: {self.get_xyz.shape[0]}')

        # Clone and Split
        extent = self.get_extent()
        # extent = self.sphere_radius
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        # Prune points below opacity
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        print(f'Prune points below min_opactiy: {prune_mask.sum()}')
        self.scalar_dict['points_below_min_opacity'] = prune_mask.sum().item()
                
        # Prune big points in world space 
        if prune_big_points:
            # Prune big points in world space
            extent = self.get_extent()
            # extent = self.sphere_radius
            big_points_ws = torch.max(self.get_scaling, dim=1).values > extent * self.percent_big_ws
            
            # Prune points too near to sphere center
            # repeat_num = 2
            # stds = self.get_scaling
            # stds = stds[:, None, :].expand(-1, repeat_num, -1) # [N, M, 1] 
            # means = torch.zeros_like(self.get_xyz)
            # means = means[:, None, :].expand(-1, repeat_num, -1) # [N, M, 3]
            # samples = torch.normal(mean=means, std=stds) # [N, M, 3]
            # rots = quaternion_to_matrix(self.get_rotation) # [N, 3, 3]
            # rots = rots[:, None, :, :].expand(-1, repeat_num, -1, -1) # [N, M, 3, 3]
            # origins = self.get_xyz[:, None, :].expand(-1, repeat_num, -1) # [N, M, 3]
            
            # samples_xyz = torch.matmul(rots, samples.unsqueeze(-1)).squeeze(-1) + origins # [N, M, 3]
            # dists = torch.linalg.norm(samples_xyz - self.sphere_center, dim=2) # [N, M]

            # initalized at 2.5r, should not be smaller than 2r
            # points_near_sphere = dists.min(dim=1).values < self.sphere_radius * 2
            
            prune_mask = torch.logical_or(prune_mask, big_points_ws)
            # prune_mask = torch.logical_or(prune_mask, points_near_sphere)
            
            # print(f'Prune points near sphere center: {points_near_sphere.sum()}')

        self.scalar_dict['points_pruned'] = prune_mask.sum().item()

        self.prune_points(prune_mask)
        
        # Reset 
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        torch.cuda.empty_cache()
        
        return self.scalar_dict, self.tensor_dict