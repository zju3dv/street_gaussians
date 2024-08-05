import torch
import torch.nn as nn
from lib.config import cfg
from lib.utils.camera_utils import Camera
from lib.utils.general_utils import get_expon_lr_func, matrix_to_axis_angle

class ColorCorrection(nn.Module):
    def __init__(self, metadata):
        super().__init__()
        self.identity_matrix = torch.eye(4).float().cuda()[:3] # [3, 4]
        
        self.config = cfg.model.color_correction
        self.mode = self.config.mode
        
        # per image embedding
        if self.mode == 'image':
            num_corrections = metadata['num_images']
        # per sensor embedding
        elif self.mode == 'sensor':
            num_corrections = metadata['num_cams']
        else:
            raise ValueError(f'Invalid mode: {self.mode}')

        if self.config.use_mlp:
            input_ch = 6
            dim = 64
            self.affine_trans = nn.Sequential( 
                nn.Linear(input_ch, dim),
                nn.ReLU(),
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, 12),
            )
            self.affine_trans[6].weight.data.fill_(0)
            self.affine_trans[6].bias.data.fill_(0)
            self.affine_trans.cuda()
            self.affine_trans_sky = nn.Sequential( 
                nn.Linear(input_ch, dim),
                nn.ReLU(),
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, 12),
            )
            self.affine_trans_sky[6].weight.data.fill_(0)
            self.affine_trans_sky[6].bias.data.fill_(0)
            self.affine_trans_sky.cuda()
        else:
            self.affine_trans = nn.Parameter(torch.eye(4).float().cuda()[:3].unsqueeze(0).repeat(num_corrections, 1, 1)).requires_grad_(True)
            self.affine_trans_sky = nn.Parameter(torch.eye(4).float().cuda()[:3].unsqueeze(0).repeat(num_corrections, 1, 1)).requires_grad_(True)
        
        self.cur_affine_trans = None

    def save_state_dict(self, is_final):
        state_dict = dict()
        state_dict['params'] = self.state_dict()
        if not is_final:
            state_dict['optimizer'] = self.optimizer.state_dict()
        return state_dict
        
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict['params'])
        if cfg.mode == 'train' and 'optimizer' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])

    def training_setup(self):
        args = cfg.optim
        color_correction_lr_init = args.get('color_correction_lr_init', 5e-4)
        color_correction_lr_final = args.get('color_correction_lr_final', 5e-5)
        color_correction_max_steps = args.get('color_correction_max_steps', cfg.train.iterations)
        if self.config.use_mlp:
            params = [
                {'params': list(self.affine_trans.parameters()), 'lr': color_correction_lr_init, 'name': 'affine_trans'},
                {'params': list(self.affine_trans_sky.parameters()), 'lr': color_correction_lr_init, 'name': 'affine_trans_sky'},
            ]
        else:
            params = [
                {'params': [self.affine_trans], 'lr': color_correction_lr_init, 'name': 'affine_trans'},
                {'params': [self.affine_trans_sky], 'lr': color_correction_lr_init, 'name': 'affine_trans_sky'},
            ]
        self.optimizer = torch.optim.Adam(params=params, lr=0, eps=1e-15)

        self.color_correction_scheduler_args = get_expon_lr_func(
            lr_init=color_correction_lr_init,
            lr_final=color_correction_lr_final,
            max_steps=color_correction_max_steps,
        )

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            lr = self.color_correction_scheduler_args(iteration)
            param_group['lr'] = lr
    
    def update_optimizer(self):
        self.optimizer.step()       
        self.optimizer.zero_grad(set_to_none=None)
        
    def get_id(self, camera: Camera):
        if self.mode == 'image':
            return camera.id
        elif self.mode == 'sensor':
            return camera.meta['cam']
        else:
            raise ValueError(f'invalid mode: {self.mode}')
        
    def get_affine_trans(self, camera: Camera, use_sky=False):
        if self.config.use_mlp:
            c2w = camera.ego_pose @ camera.extrinsic
            c2w = matrix_to_axis_angle(c2w.unsqueeze(0)).squeeze(0)
            if use_sky:
                affine_trans = self.affine_trans_sky(c2w).view(3, 4) + self.identity_matrix
            else:
                affine_trans = self.affine_trans(c2w).view(3, 4) + self.identity_matrix
            
        else:
            id = self.get_id(camera)
            if use_sky:
                affine_trans = self.affine_trans_sky[id]
            else:
                affine_trans = self.affine_trans[id]
                        
        self.cur_affine_trans = affine_trans
            
        return affine_trans
                
    def forward(self, camera: Camera, image: torch.Tensor, use_sky=False):
        affine_trans = self.get_affine_trans(camera, use_sky)
        image = torch.einsum('ij, jhw -> ihw', affine_trans[:3, :3], image) + affine_trans[:3, 3].unsqueeze(-1).unsqueeze(-1)
        return image

    def regularization_loss(self, camera: Camera):
        affine_trans = self.get_affine_trans(camera, use_sky=False)
        affine_trans_sky = self.get_affine_trans(camera, use_sky=True)
        
        loss = torch.abs(affine_trans - self.identity_matrix) + torch.abs(affine_trans_sky - self.identity_matrix)
        loss = loss.mean()
        return loss

