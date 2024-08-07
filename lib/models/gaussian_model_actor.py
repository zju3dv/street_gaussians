import torch
import torch.nn as nn
import numpy as np
import os
from lib.config import cfg
from lib.models.gaussian_model import GaussianModel
from lib.utils.general_utils import quaternion_to_matrix, inverse_sigmoid, matrix_to_quaternion, get_expon_lr_func, quaternion_raw_multiply
from lib.utils.sh_utils import RGB2SH, IDFT
from lib.datasets.base_readers import fetchPly
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2

class GaussianModelActor(GaussianModel):
    def __init__(
        self, 
        model_name, 
        obj_meta, 
    ):
        # parse obj_meta
        self.obj_meta = obj_meta
        
        self.obj_class = obj_meta['class']
        self.obj_class_label = obj_meta['class_label']
        self.deformable = obj_meta['deformable']         
        self.start_frame = obj_meta['start_frame']
        self.start_timestamp = obj_meta['start_timestamp']
        self.end_frame = obj_meta['end_frame']
        self.end_timestamp = obj_meta['end_timestamp']
        self.track_id = obj_meta['track_id']
        
        # fourier spherical harmonics
        self.fourier_dim = cfg.model.gaussian.get('fourier_dim', 1)
        self.fourier_scale = cfg.model.gaussian.get('fourier_scale', 1.)
        
        # bbox
        length, width, height = obj_meta['length'], obj_meta['width'], obj_meta['height']
        self.bbox = np.array([length, width, height]).astype(np.float32)
        xyz = torch.tensor(self.bbox).float().cuda()
        self.min_xyz, self.max_xyz =  -xyz/2., xyz/2.  
        
        extent = max(length*1.5/cfg.data.box_scale, width*1.5/cfg.data.box_scale, height) / 2.
        self.extent = torch.tensor([extent]).float().cuda()   

        num_classes = 1 if cfg.data.get('use_semantic', False) else 0
        self.num_classes_global = cfg.data.num_classes if cfg.data.get('use_semantic', False) else 0        
        super().__init__(model_name=model_name, num_classes=num_classes)
        
        self.flip_prob = cfg.model.gaussian.get('flip_prob', 0.) if not self.deformable else 0.
        self.flip_axis = 1 

        self.spatial_lr_scale = extent

    def get_extent(self):
        max_scaling = torch.max(self.get_scaling, dim=1).values

        extent_lower_bound = torch.topk(max_scaling, int(self.get_xyz.shape[0] * 0.1), largest=False).values[-1] / self.percent_dense
        extent_upper_bound = torch.topk(max_scaling, int(self.get_xyz.shape[0] * 0.1), largest=True).values[-1] / self.percent_dense
        
        extent = torch.clamp(self.extent, min=extent_lower_bound, max=extent_upper_bound)        
        print(f'extent: {extent.item()}, extent bound: [{extent_lower_bound}, {extent_upper_bound}]')

        return extent
    @property
    def get_semantic(self):
        semantic = torch.zeros((self.get_xyz.shape[0], self.num_classes_global)).float().cuda()
        if self.semantic_mode == 'logits':
            semantic[:, self.obj_class_label] = self._semantic[:, 0] # ubounded semantic        
        elif self.semantic_mode == 'probabilities':
            semantic[:, self.obj_class_label] = torch.nn.functional.sigmoid(self._semantic[:, 0]) # 0 ~ 1

        return semantic 

    def get_features_fourier(self, frame=0):
        normalized_frame = (frame - self.start_frame) / (self.end_frame - self.start_frame)
        time = self.fourier_scale * normalized_frame

        idft_base = IDFT(time, self.fourier_dim)[0].cuda()
        features_dc = self._features_dc # [N, C, 3]
        features_dc = torch.sum(features_dc * idft_base[..., None], dim=1, keepdim=True) # [N, 1, 3]
        features_rest = self._features_rest # [N, sh, 3]
        features = torch.cat([features_dc, features_rest], dim=1) # [N, (sh + 1) * C, 3]
        return features
           
    def create_from_pcd(self, spatial_lr_scale):
        pointcloud_path = os.path.join(cfg.model_path, 'input_ply', f'points3D_{self.model_name}.ply')   
        if os.path.exists(pointcloud_path):
            pcd = fetchPly(pointcloud_path)
            pointcloud_xyz = np.asarray(pcd.points)
            if pointcloud_xyz.shape[0] < 2000:
                self.random_initialization = True
            else:
                self.random_initialization = False
        else:
            self.random_initialization = True

        if self.random_initialization is True:
            points_dim = 20
            print(f'Creating random pointcloud for {self.model_name}')
            points_x, points_y, points_z = np.meshgrid(
                np.linspace(-1., 1., points_dim), np.linspace(-1., 1., points_dim), np.linspace(-1., 1., points_dim),
            )
            
            points_x = points_x.reshape(-1)
            points_y = points_y.reshape(-1)
            points_z = points_z.reshape(-1)

            bbox_xyz_scale = self.bbox / 2.
            pointcloud_xyz = np.stack([points_x, points_y, points_z], axis=-1)
            pointcloud_xyz = pointcloud_xyz * bbox_xyz_scale            
            pointcloud_rgb = np.random.rand(*pointcloud_xyz.shape).astype(np.float32)  
        elif not self.deformable and self.flip_prob > 0.:          
            pcd = fetchPly(pointcloud_path)
            pointcloud_xyz = np.asarray(pcd.points)
            pointcloud_rgb = np.asarray(pcd.colors)
            num_pointcloud_1 = (pointcloud_xyz[:, self.flip_axis] > 0).sum()
            num_pointcloud_2 = (pointcloud_xyz[:, self.flip_axis] < 0).sum()
            if num_pointcloud_1 >= num_pointcloud_2:
                pointcloud_xyz_part = pointcloud_xyz[pointcloud_xyz[:, self.flip_axis] > 0]
                pointcloud_rgb_part = pointcloud_rgb[pointcloud_xyz[:, self.flip_axis] > 0]
            else:
                pointcloud_xyz_part = pointcloud_xyz[pointcloud_xyz[:, self.flip_axis] < 0]
                pointcloud_rgb_part = pointcloud_rgb[pointcloud_xyz[:, self.flip_axis] < 0]
            pointcloud_xyz_flip = pointcloud_xyz_part.copy()
            pointcloud_xyz_flip[:, self.flip_axis] *= -1
            pointcloud_rgb_flip = pointcloud_rgb_part.copy()
            pointcloud_xyz = np.concatenate([pointcloud_xyz, pointcloud_xyz_flip], axis=0)
            pointcloud_rgb = np.concatenate([pointcloud_rgb, pointcloud_rgb_flip], axis=0)
        else:
            pcd = fetchPly(pointcloud_path)
            pointcloud_xyz = np.asarray(pcd.points)
            pointcloud_rgb = np.asarray(pcd.colors)
            
        fused_point_cloud = torch.tensor(np.asarray(pointcloud_xyz)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pointcloud_rgb)).float().cuda())

        # features = torch.zeros((fused_color.shape[0], 3, 
        #                         (self.max_sh_degree + 1) ** 2 * self.fourier_dim)).float().cuda()
        # features[:, :3, 0] = fused_color
        features_dc = torch.zeros((fused_color.shape[0], 3, self.fourier_dim)).float().cuda()
        features_rest = torch.zeros(fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1).float().cuda()
        features_dc[:, :3, 0] = fused_color

        print(f"Number of points at initialisation for {self.model_name}: ", fused_point_cloud.shape[0])
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pointcloud_xyz)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4)).cuda()
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1))).float().cuda()
        semantics = torch.zeros((fused_point_cloud.shape[0], self.num_classes)).float().cuda()
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        
        # self._features_dc = nn.Parameter(features[:, :, :self.fourier_dim].transpose(1, 2).contiguous().requires_grad_(True))
        # self._features_rest = nn.Parameter(features[:, :, self.fourier_dim:].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_dc = nn.Parameter(features_dc.transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features_rest.transpose(1, 2).contiguous().requires_grad_(True))
        
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._semantic = nn.Parameter(semantics.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")


    def training_setup(self):
        args = cfg.optim

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 2), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.active_sh_degree = 0
                
        tag = 'obj'
        position_lr_init = args.get('position_lr_init_{}'.format(tag), args.position_lr_init)
        position_lr_final = args.get('position_lr_final_{}'.format(tag), args.position_lr_final)
        scaling_lr = args.get('scaling_lr_{}'.format(tag), args.scaling_lr)
        feature_lr = args.get('feature_lr_{}'.format(tag), args.feature_lr)
        semantic_lr = args.get('semantic_lr_{}'.format(tag), args.semantic_lr)
        rotation_lr = args.get('rotation_lr_{}'.format(tag), args.rotation_lr)
        opacity_lr = args.get('opacity_lr_{}'.format(tag), args.opacity_lr)
        feature_rest_lr = args.get('feature_rest_lr_{}'.format(tag), feature_lr / 20.0)

        l = [
            {'params': [self._xyz], 'lr': position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': feature_rest_lr, "name": "f_rest"},
            {'params': [self._opacity], 'lr': opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': rotation_lr, "name": "rotation"},
            {'params': [self._semantic], 'lr': semantic_lr, "name": "semantic"},
        ]
        
        self.percent_dense = args.percent_dense
        self.percent_big_ws = args.percent_big_ws
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=position_lr_init * self.spatial_lr_scale,
            lr_final=position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=args.position_lr_delay_mult,
            max_steps=args.position_lr_max_steps
        )
        
        self.densify_and_prune_list = ['xyz, f_dc, f_rest, opacity, scaling, rotation, semantic']
        self.scalar_dict = dict()
        self.tensor_dict = dict()  
            
    def densify_and_prune(self, max_grad, min_opacity, prune_big_points):
        if not (self.random_initialization or self.deformable):
            max_grad = cfg.optim.get('densify_grad_threshold_obj', max_grad)
            if cfg.optim.get('densify_grad_abs_obj', False):
                grads = self.xyz_gradient_accum[:, 1:2] / self.denom
            else:
                grads = self.xyz_gradient_accum[:, 0:1] / self.denom
        else:
            grads = self.xyz_gradient_accum[:, 0:1] / self.denom
        
        grads[grads.isnan()] = 0.0

        # Clone and Split
        # extent = self.get_extent()
        extent = self.extent
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        # Prune points below opacity
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        
        if prune_big_points:
            # Prune big points in world space
            extent = self.extent
            big_points_ws = self.get_scaling.max(dim=1).values > extent * self.percent_big_ws
            
            # Prune points outside the tracking box
            repeat_num = 2
            stds = self.get_scaling
            stds = stds[:, None, :].expand(-1, repeat_num, -1) # [N, M, 1] 
            means = torch.zeros_like(self.get_xyz)
            means = means[:, None, :].expand(-1, repeat_num, -1) # [N, M, 3]
            samples = torch.normal(mean=means, std=stds) # [N, M, 3]
            rots = quaternion_to_matrix(self.get_rotation) # [N, 3, 3]
            rots = rots[:, None, :, :].expand(-1, repeat_num, -1, -1) # [N, M, 3, 3]
            origins = self.get_xyz[:, None, :].expand(-1, repeat_num, -1) # [N, M, 3]
                        
            samples_xyz = torch.matmul(rots, samples.unsqueeze(-1)).squeeze(-1) + origins # [N, M, 3]                    
            num_gaussians = self.get_xyz.shape[0]
            points_inside_box = torch.logical_and(
                torch.all((samples_xyz >= self.min_xyz).view(num_gaussians, -1), dim=-1),
                torch.all((samples_xyz <= self.max_xyz).view(num_gaussians, -1), dim=-1),
            )
            points_outside_box = torch.logical_not(points_inside_box)           
            
            prune_mask = torch.logical_or(prune_mask, big_points_ws)
            prune_mask = torch.logical_or(prune_mask, points_outside_box)
            
        self.prune_points(prune_mask)
        
        # Reset
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 2), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        torch.cuda.empty_cache()
        
        return self.scalar_dict, self.tensor_dict
    
    def set_max_radii(self, visibility_obj, max_radii2D):
        self.max_radii2D[visibility_obj] = torch.max(self.max_radii2D[visibility_obj], max_radii2D[visibility_obj])
    
    def box_reg_loss(self):
        scaling_max = self.get_scaling.max(dim=1).values
        scaling_max = torch.where(scaling_max > self.extent * self.percent_dense, scaling_max, 0.)
        reg_loss = (scaling_max / self.extent).mean()
        
        return reg_loss
        
        
    
    