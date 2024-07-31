import torch
import numpy as np
import os
from lib.config import cfg
from lib.utils.graphics_utils import BasicPointCloud
from lib.datasets.base_readers import fetchPly
from lib.models.gaussian_model import GaussianModel
from lib.utils.camera_utils import Camera, make_rasterizer


class GaussianModelBkgd(GaussianModel):
    def __init__(
        self, 
        model_name='background', 
        scene_center=np.array([0, 0, 0]),
        scene_radius=20,
        sphere_center=np.array([0, 0, 0]),
        sphere_radius=20,
    ):
        self.scene_center = torch.from_numpy(scene_center).float().cuda()
        self.scene_radius = torch.tensor([scene_radius]).float().cuda()
        self.sphere_center = torch.from_numpy(sphere_center).float().cuda()
        self.sphere_radius = torch.tensor([sphere_radius]).float().cuda()
        num_classes = cfg.data.num_classes if cfg.data.get('use_semantic', False) else 0
        self.background_mask = None

        super().__init__(model_name=model_name, num_classes=num_classes)

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float): 
        print('Create background model')
        # pointcloud_path_sky =  os.path.join(cfg.model_path, 'input_ply', 'points3D_sky.ply')
        # include_sky = cfg.model.nsg.get('include_sky', False)
        # if os.path.exists(pointcloud_path_sky) and not include_sky:
        #     pcd_sky = fetchPly(pointcloud_path_sky)
        #     pointcloud_xyz = np.concatenate((pcd.points, pcd_sky.points), axis=0)
        #     pointcloud_rgb = np.concatenate((pcd.colors, pcd_sky.colors), axis=0)
        #     pointcloud_normal = np.zeros_like(pointcloud_xyz)
        #     pcd = BasicPointCloud(pointcloud_xyz, pointcloud_rgb, pointcloud_normal)
        return super().create_from_pcd(pcd, spatial_lr_scale)

    def set_background_mask(self, camera: Camera):
        pass
    
    @property
    def get_scaling(self):
        scaling = super().get_scaling
        return scaling if self.background_mask is None else scaling[self.background_mask]

    @property
    def get_rotation(self):
        rotation = super().get_rotation
        return rotation if self.background_mask is None else rotation[self.background_mask]

    @property
    def get_xyz(self):
        xyz = super().get_xyz
        return xyz if self.background_mask is None else xyz[self.background_mask]        
    
    @property
    def get_features(self):
        features = super().get_features
        return features if self.background_mask is None else features[self.background_mask]        
    
    @property
    def get_opacity(self):
        opacity = super().get_opacity
        return opacity if self.background_mask is None else opacity[self.background_mask]
    
    @property
    def get_semantic(self):
        semantic = super().get_semantic
        return semantic if self.background_mask is None else semantic[self.background_mask]

    def densify_and_prune(self, max_grad, min_opacity, prune_big_points):
        max_grad = cfg.optim.get('densify_grad_threshold_bkgd', max_grad)
        if cfg.optim.get('densify_grad_abs_bkgd', False):
            grads = self.xyz_gradient_accum[:, 1:2] / self.denom
        else:
            grads = self.xyz_gradient_accum[:, 0:1] / self.denom
        grads[grads.isnan()] = 0.0
        self.scalar_dict.clear()
        self.tensor_dict.clear()    
        self.scalar_dict['points_total'] = self.get_xyz.shape[0]

        # Clone and Split
        extent = self.scene_radius
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        # Prune points below opacity
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        self.scalar_dict['points_below_min_opacity'] = prune_mask.sum().item()

        # Prune big points in world space 
        if prune_big_points:
            dists = torch.linalg.norm(self.get_xyz - self.sphere_center, dim=1)            
            big_points_ws = torch.max(self.get_scaling, dim=1).values > extent * self.percent_big_ws
            big_points_ws[dists > 2 * self.sphere_radius] = False
            
            prune_mask = torch.logical_or(prune_mask, big_points_ws)
            
            self.scalar_dict['points_big_ws'] = big_points_ws.sum().item()

        self.scalar_dict['points_pruned'] = prune_mask.sum().item()
        self.prune_points(prune_mask)
        
        # Reset 
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 2), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        torch.cuda.empty_cache()

        return self.scalar_dict, self.tensor_dict