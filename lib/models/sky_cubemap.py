import torch
import torch.nn as nn
import numpy as np
import imageio
import os
import cv2
import nvdiffrast.torch as dr 
from lib.config import cfg
from lib.utils.camera_utils import Camera
from lib.utils.graphics_utils import get_rays_torch
from lib.utils.general_utils import get_expon_lr_func
from sklearn.cluster import KMeans

class SkyCubeMap(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.cfg = cfg.model.sky
        self.resolution = self.cfg.resolution
        eps = 1e-3
        
        if self.cfg.white_background:
            base = torch.ones(6, self.resolution, self.resolution, 3).float().cuda() * (1. - eps)
        else:
            base = torch.zeros(6, self.resolution, self.resolution, 3).float().cuda() + eps
            
        self.sky_cube_map = nn.Parameter(base).requires_grad_(True)    
        
        # # TODO: change hard code here
        max_h, max_w = 1080, 1920
        if cfg.data.white_background:
            self.sky_color = torch.ones((max_h, max_w, 3)).float().cuda()
        else:
            self.sky_color = torch.zeros((max_h, max_w, 3)).float().cuda()    
        
    def save_state_dict(self, is_final):
        state_dict = dict()
        state_dict['params'] = self.state_dict()
        if not is_final: 
            state_dict['optimizer'] = self.optimizer.state_dict()
        
        # save LDR cubemap now
        sky_latlong = cubemap_to_latlong(self.sky_cube_map, [self.resolution, self.resolution * 2])
        sky_latlong = (sky_latlong.clamp(0., 1.).detach().cpu().numpy() * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(cfg.model_path, 'sky_latlong.png'), sky_latlong)
        return state_dict
        
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict['params'])
        if cfg.mode == 'train' and 'optimizer' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])
    
    def training_setup(self):
        args = cfg.optim
        sky_cube_map_lr_init = args.get('sky_cube_map_lr_init', 0.01)
        sky_cube_map_lr_final = args.get('sky_cube_map_lr_final', 0.0001) 
        sky_cube_map_max_steps = args.get('sky_cube_map_max_steps', cfg.train.iterations)
        params = [{'params': [self.sky_cube_map], 'lr': sky_cube_map_lr_init, 'name':'sky_cube_map'}]
        self.optimizer = torch.optim.Adam(params=params, lr=0, eps=1e-15)

        self.sky_cube_map_scheduler_args = get_expon_lr_func(
            lr_init=sky_cube_map_lr_init,
            lr_final=sky_cube_map_lr_final,
            max_steps=sky_cube_map_max_steps,
        )
          
    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "sky_cube_map":
                lr = self.sky_cube_map_scheduler_args(iteration)
                param_group['lr'] = lr
    
    def update_optimizer(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=None)

    def forward(self, camera: Camera, acc=None):
        # acc: gaussian opacity of foreground model: [1, H, W]
        # mask: [H, W], indicating whether a pixel is covered by forgeground model
        if cfg.mode == 'train' and hasattr(camera, 'original_sky_mask'):
            mask = camera.original_sky_mask.cuda()[0]
            mask[:50, :] = True
        elif acc is not None:
            mask = (1 - acc[0]) > 1e-3
        else:
            mask = None

        # R, T should be in w2c format
        # rays_d: [H, W, 3]
        w2c = camera.world_view_transform.transpose(0, 1)
        H, W, K, R, T = camera.image_height, camera.image_width, camera.K, w2c[:3, :3], w2c[:3, 3]
        if cfg.mode == 'train':
            _, rays_d = get_rays_torch(H, W, K, R, T, perturb=True)
        else:
            _, rays_d = get_rays_torch(H, W, K, R, T, perturb=False)
        
        if mask is None:
            sky_color = dr.texture(self.sky_cube_map[None, ...], rays_d[None, ...],
                               filter_mode='linear', boundary_mode='cube')
            sky_color = sky_color[0].permute(2, 0, 1).clamp(0., 1.) # [3, H, W]
        else:
            if cfg.mode == 'train':
                if self.cfg.white_background:
                    sky_color = torch.ones((H, W, 3)).float().cuda() 
                else:
                    sky_color = torch.zeros((H, W, 3)).float().cuda() 
            else:
                sky_color = self.sky_color[:H, :W, :]
                if self.cfg.white_background:
                    torch.fill_(sky_color, 1.)
                else:
                    torch.fill_(sky_color, 0.)

            rays_d = rays_d[mask] # [N, 3]
            sky_color_mask = dr.texture(self.sky_cube_map[None, ...], rays_d[None, None, ...],
                               filter_mode='linear', boundary_mode='cube')

            sky_color_mask = sky_color_mask.squeeze(0).squeeze(0)                 
            sky_color[mask] = sky_color_mask
            sky_color = sky_color.permute(2, 0, 1).clamp(0., 1.) # [3, H, W]
                
        return sky_color

#----------------------------------------------------------------------------
# Vector operations
#----------------------------------------------------------------------------

def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def reflect(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return 2*dot(x, n)*n - x

def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)

def to_hvec(x: torch.Tensor, w: float) -> torch.Tensor:
    return torch.nn.functional.pad(x, pad=(0,1), mode='constant', value=w)

#----------------------------------------------------------------------------
# Cubemap utility functions
#----------------------------------------------------------------------------

def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)

def cube_to_dir(s, x, y):
    if s == 0:   rx, ry, rz = torch.ones_like(x), -y, -x
    elif s == 1: rx, ry, rz = -torch.ones_like(x), -y, x
    elif s == 2: rx, ry, rz = x, torch.ones_like(x), y
    elif s == 3: rx, ry, rz = x, -torch.ones_like(x), -y
    elif s == 4: rx, ry, rz = x, -y, torch.ones_like(x)
    elif s == 5: rx, ry, rz = -x, -y, -torch.ones_like(x)
    return torch.stack((rx, ry, rz), dim=-1)

def latlong_to_cubemap(latlong_map, res):
    cubemap = torch.zeros(6, res[0], res[1], latlong_map.shape[-1], dtype=torch.float32, device='cuda')
    for s in range(6):
        gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device='cuda'), 
                                torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device='cuda'),
                                indexing='ij')
        v = safe_normalize(cube_to_dir(s, gx, gy))

        tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5
        tv = torch.acos(torch.clamp(v[..., 1:2], min=-1, max=1)) / np.pi
        texcoord = torch.cat((tu, tv), dim=-1)

        cubemap[s, ...] = dr.texture(latlong_map[None, ...], texcoord[None, ...], filter_mode='linear')[0]
    return cubemap

def cubemap_to_latlong(cubemap, res):
    gy, gx = torch.meshgrid(torch.linspace( 0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device='cuda'), 
                            torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device='cuda'),
                            indexing='ij')
    
    sintheta, costheta = torch.sin(gy*np.pi), torch.cos(gy*np.pi)
    sinphi, cosphi     = torch.sin(gx*np.pi), torch.cos(gx*np.pi)
    
    reflvec = torch.stack((
        sintheta*sinphi, 
        costheta, 
        -sintheta*cosphi
        ), dim=-1)
    return dr.texture(cubemap[None, ...], reflvec[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')[0]

from lib.utils.vq_utils import vq, vq_st
class SkyCubeMap_codebook(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.cfg = cfg.model.sky
        self.resolution = self.cfg.resolution
        
        feat_dim = 12
        codebook_size = 32
        
        self.sky_codebook = nn.Parameter(torch.zeros(codebook_size, feat_dim).float().cuda()).requires_grad_(True)
        
        if cfg.mode == 'train':
            self.sky_cube_map = nn.Parameter(torch.zeros(6, self.resolution, self.resolution, feat_dim).float().cuda()).requires_grad_(True)
        else:
            self.sky_cube_map = nn.Parameter(torch.zeros(6, self.resolution, self.resolution, 1).float().cuda())

        input_dim = feat_dim + 3
        self.mlp_shader = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        ).cuda()
        
        # # TODO: change hard code here
        max_h, max_w = 1080, 1920
        if cfg.data.white_background:
            self.sky_color = torch.ones((max_h, max_w, 3)).float().cuda()
        else:
            self.sky_color = torch.zeros((max_h, max_w, 3)).float().cuda()    

    def save_state_dict(self, is_final):
        state_dict = dict()       
        if not is_final:
            state_dict['params'] = self.state_dict()
            state_dict['optimizer'] = self.optimizer.state_dict()
        else:
            params = self.state_dict()
            sky_cube_map_flatten = self.sky_cube_map.reshape(6 * self.resolution * self.resolution, -1) # [N, F]
            sky_cube_map_indices = vq(sky_cube_map_flatten, self.sky_codebook) # [N]
            #TODO: compress to smaller size: log2(codebook size) bit
            sky_cube_map_indices = sky_cube_map_indices.reshape(6, self.resolution, self.resolution, 1)
            params['sky_cube_map'] = sky_cube_map_indices
            state_dict['params'] = params
                    
        return state_dict
        
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict['params'], strict=False)
        if cfg.mode == 'train' and 'optimizer' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])
    
    def training_setup(self):
        args = cfg.optim
        
        sky_cube_map_lr_init = args.get('sky_cube_map_lr', 0.01)
        sky_cube_map_lr_final = args.get('sky_cube_map_lr_final', 0.0001) 

        mlp_shader_lr_init = args.get('mlp_shader_lr_init', 5e-4)
        mlp_shader_lr_final = args.get('mlp_shader_lr_final', 1e-5)
        
        max_steps = args.get('sky_cube_map_max_steps', cfg.train.iterations)
        
        params = [{'params': [self.sky_cube_map], 'lr': sky_cube_map_lr_init, 'name': 'sky_cube_map'},
                  {'params': [self.sky_codebook], 'lr': sky_cube_map_lr_init, 'name': 'sky_codebook'},
                  {'params': list(self.mlp_shader.parameters()), 'lr': mlp_shader_lr_init, 'name': 'mlp_shader'}]
        
        self.optimizer = torch.optim.Adam(params=params, lr=0, eps=1e-15)
        
        self.sky_cube_map_scheduler_args = get_expon_lr_func(
            lr_init=sky_cube_map_lr_init,
            lr_final=sky_cube_map_lr_final,
            max_steps=max_steps,
        )
        
        self.mlp_shader_scheduler_args = get_expon_lr_func(
            lr_init=mlp_shader_lr_init,
            lr_final=mlp_shader_lr_final,
            max_steps=max_steps,
        )
          
    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "mlp_shader":
                lr = self.mlp_shader_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "sky_cube_map" or 'sky_codebook':
                lr = self.sky_cube_map_scheduler_args(iteration)
                param_group['lr'] = lr
            
    def update_optimizer(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=None)
        
    def forward_train(self, camera: Camera, acc):       
        # acc: gaussian opacity of foreground model: [1, H, W]
        # mask: [H, W], indicating whether a pixel is covered by forgeground model
        if hasattr(camera, 'original_sky_mask'):
            mask = camera.original_sky_mask.cuda()[0]
            mask[:50, :] = True
        elif acc is not None:
            mask = (1 - acc[0]) > 1e-3
        else:
            mask = None
         
        mask = camera.original_sky_mask.cuda()[0]
        mask[:50, :] = True
                
        # R, T should be in w2c format
        # rays_d: [H, W, 3]
        w2c = camera.world_view_transform.transpose(0, 1)
        H, W, K, R, T = camera.image_height, camera.image_width, camera.K, w2c[:3, :3], w2c[:3, 3]
        _, rays_d = get_rays_torch(H, W, K, R, T, perturb=False)
        
        if cfg.mode == 'train':
            if self.cfg.white_background:
                sky_color = torch.ones((H, W, 3)).float().cuda() 
            else:
                sky_color = torch.zeros((H, W, 3)).float().cuda() 
        
        rays_d = rays_d[mask] # [N, 3]
        feat = dr.texture(self.sky_cube_map[None, ...], rays_d[None, None, ...], 
                          filter_mode='linear', boundary_mode='cube') 
        
        feat_enc = feat.squeeze(0).squeeze(0) # [N, F]
        feat_dec_detach, indices = vq_st(feat_enc, self.sky_codebook.detach())
        feat_dec = self.sky_codebook[indices]

        shader_input = torch.cat([feat_dec_detach, rays_d], dim=-1)
        sky_color_mask = self.mlp_shader(shader_input)
        sky_color_mask = torch.sigmoid(sky_color_mask)
            
        sky_color[mask] = sky_color_mask
        sky_color = sky_color.permute(2, 0, 1).clamp(0., 1.) # [3, H, W]
        
        self.vq_loss_cache = {
            'feat_enc': feat_enc,
            'feat_dec': feat_dec,
        }
            
        return sky_color
    
    def forward_inference(self, camera: Camera, acc):
        mask = (1 - acc[0]) > 0
        # R, T should be in w2c format
        # rays_d: [H, W, 3]
        w2c = camera.world_view_transform.transpose(0, 1)
        H, W, K, R, T = camera.image_height, camera.image_width, camera.K, w2c[:3, :3], w2c[:3, 3]
        _, rays_d = get_rays_torch(H, W, K, R, T, perturb=False)
        sky_color = self.sky_color[:H, :W, :]
        if self.cfg.white_background:
            torch.fill_(sky_color, 1.)
        else:
            torch.fill_(sky_color, 0.)
            
        rays_d = rays_d[mask] # [N, 3]
        feat_indices = dr.texture(self.sky_cube_map[None, ...], rays_d[None, None, ...], 
                          filter_mode='linear', boundary_mode='cube') 
        feat_indices = feat_indices.squeeze(0).squeeze(0).squeeze(-1).long()
        
        # feat_enc = feat.squeeze(0).squeeze(0) # [N, F]
        # indices = vq(feat_enc, self.sky_codebook)
        feat_dec = self.sky_codebook[feat_indices]
        
        shader_input = torch.cat([feat_dec, rays_d], dim=-1)
        sky_color_mask = self.mlp_shader(shader_input)
        sky_color_mask = torch.sigmoid(sky_color_mask)
            
        sky_color[mask] = sky_color_mask
        sky_color = sky_color.permute(2, 0, 1).clamp(0., 1.) # [3, H, W]
                
        return sky_color
        
    def forward(self, camera: Camera, acc):
        # acc: gaussian opacity of foreground model: [1, H, W]
        # mask: [H, W], indicating whether a pixel is covered by forgeground model
        if cfg.mode == 'train':
            sky_color = self.forward_train(camera)
        else:
            sky_color = self.forward_inference(camera, acc)
            
        return sky_color
