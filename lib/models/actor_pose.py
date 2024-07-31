import torch
import torch.nn as nn
import numpy as np
from lib.utils.general_utils import quaternion_raw_multiply, get_expon_lr_func, quaternion_slerp, matrix_to_quaternion
from lib.config import cfg
from lib.utils.camera_utils import Camera

class ActorPose(nn.Module):      
    def __init__(self, tracklets, tracklet_timestamps, camera_timestamps, obj_info):
        # tracklets: [num_frames, max_obj, [track_id, x, y, z, qw, qx, qy, qz]]
        # frame_timestamps: [num_frames]
        super().__init__()
        tracklets = torch.from_numpy(tracklets).float().cuda()
        self.track_ids = tracklets[..., 0] # [num_frames, max_obj]
        self.input_trans = tracklets[..., 1:4] # [num_frames, max_obj, [x, y, z]]
        self.input_rots = tracklets[..., 4:8] # [num_frames, max_obj, [qw, qx, qy, qz]]
        self.timestamps = tracklet_timestamps
        self.camera_timestamps = camera_timestamps

        self.opt_track = cfg.model.nsg.opt_track
        if self.opt_track:
            self.opt_trans = nn.Parameter(torch.zeros_like(self.input_trans)).requires_grad_(True) 
            # [num_frames, max_obj, [dx, dy, dz]]

            self.opt_rots = nn.Parameter(torch.zeros_like(self.input_rots[..., :1])).requires_grad_(True) 
            # [num_frames, max_obj, [dtheta]
        
        self.obj_info = obj_info
        for track_id in self.obj_info.keys():
            self.obj_info[track_id]['track_idx'] = torch.argwhere(self.track_ids == track_id)

    def save_state_dict(self, is_final):
        state_dict = dict()
        if self.opt_track:
            state_dict['params'] = self.state_dict()
        if not is_final:
            state_dict['optimizer'] = self.optimizer.state_dict()
        return state_dict
        
    def load_state_dict(self, state_dict):
        if self.opt_track:
            super().load_state_dict(state_dict['params'])
            if cfg.mode == 'train' and 'optimizer' in state_dict:
                self.optimizer.load_state_dict(state_dict['optimizer'])

    def training_setup(self):
        args = cfg.optim
        if self.opt_track:
            params = [
                {'params': [self.opt_trans], 'lr': args.track_position_lr_init, 'name': 'opt_trans'},
                {'params': [self.opt_rots], 'lr': args.track_rotation_lr_init, 'name': 'opt_rots'},
            ]
            
            self.opt_trans_scheduler_args = get_expon_lr_func(lr_init=args.track_position_lr_init,
                                                    lr_final=args.track_position_lr_final,
                                                    lr_delay_mult=args.track_position_lr_delay_mult,
                                                    max_steps=args.track_position_max_steps,
                                                    warmup_steps=args.opacity_reset_interval)
            
            self.opt_rots_scheduler_args = get_expon_lr_func(lr_init=args.track_rotation_lr_init,
                                                    lr_final=args.track_rotation_lr_final,
                                                    lr_delay_mult=args.track_rotation_lr_delay_mult,
                                                    max_steps=args.track_rotation_max_steps,
                                                    warmup_steps=args.opacity_reset_interval)    
            
            self.optimizer = torch.optim.Adam(params=params, lr=0, eps=1e-15)
    
    def update_learning_rate(self, iteration):
        if self.opt_track:
            for param_group in self.optimizer.param_groups:
                if param_group["name"] == "opt_trans":
                    lr = self.opt_trans_scheduler_args(iteration)
                    param_group['lr'] = lr
                if param_group["name"] == "opt_rots":
                    lr = self.opt_rots_scheduler_args(iteration)
                    param_group['lr'] = lr
        
    def update_optimizer(self):
        if self.opt_track:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=None)
        
    def find_closest_indices(self, track_id, timestamp):
        track_idx = self.obj_info[track_id]['track_idx']
        frame_idx = track_idx[:, 0].cpu()
        frame_timestamps = np.array(self.timestamps[frame_idx])
        assert len(frame_timestamps) > 1
        delta_timestamps = np.abs(frame_timestamps - timestamp)
        idx1, idx2 = np.argsort(delta_timestamps)[:2]
        return track_idx[idx1], track_idx[idx2]

    
    def find_closest_camera_timestamps(self, track_id, camera: Camera):
        timestamp = camera.meta['timestamp']
        cam = camera.meta['cam']
        camera_timestamps = self.camera_timestamps[cam]['train_timestamps']
        start_timestamp = self.obj_info[track_id]['start_timestamp']
        end_timestamp = self.obj_info[track_id]['end_timestamp']
        camera_timestamps = np.array([x for x in camera_timestamps if x >= start_timestamp and x <= end_timestamp])
        if len(camera_timestamps) < 2:
            return None, None
        else:
            delta_timestamps = np.abs(camera_timestamps - timestamp)
            idx1, idx2 = np.argsort(delta_timestamps)[:2]            
            return camera_timestamps[idx1], camera_timestamps[idx2]
            
    def get_tracking_translation_(self, track_id, timestamp):
        ind1, ind2 = self.find_closest_indices(track_id, timestamp)
        frame_ind1, frame_ind2 = ind1[0], ind2[0]
        column_ind1, column_ind2 = ind1[1], ind2[1]
        timestamp1, timestamp2 = self.timestamps[frame_ind1.cpu()], self.timestamps[frame_ind2.cpu()]

        if self.opt_track:
            trans1 = self.input_trans[frame_ind1, column_ind1] + self.opt_trans[frame_ind1, column_ind1]
            trans2 = self.input_trans[frame_ind2, column_ind2] + self.opt_trans[frame_ind2, column_ind2]
        else:
            trans1 = self.input_trans[frame_ind1, column_ind1]
            trans2 = self.input_trans[frame_ind2, column_ind2]
        
        trans = (trans1 * (timestamp2 - timestamp) + trans2 * (timestamp - timestamp1)) / (timestamp2 - timestamp1)        
            
        return trans
    
    def get_tracking_translation(self, track_id, camera: Camera):
        if self.opt_track and camera.meta['is_val']:
            timestamp1, timestamp2 = self.find_closest_camera_timestamps(track_id, camera)
            if timestamp1 is None:
                return self.get_tracking_translation_(track_id, camera.meta['timestamp'])
            else:
                timestamp = camera.meta['timestamp']
                trans1 = self.get_tracking_translation_(track_id, timestamp1)
                trans2 = self.get_tracking_translation_(track_id, timestamp2)
                trans = (trans1 * (timestamp2 - timestamp) + trans2 * (timestamp - timestamp1)) / (timestamp2 - timestamp1)
                return trans
        else:
            return self.get_tracking_translation_(track_id, camera.meta['timestamp'])

    def get_tracking_rotation_(self, track_id, timestamp):
        ind1, ind2 = self.find_closest_indices(track_id, timestamp)
        frame_ind1, frame_ind2 = ind1[0], ind2[0]
        column_ind1, column_ind2 = ind1[1], ind2[1]
        timestamp1, timestamp2 = self.timestamps[frame_ind1.cpu()], self.timestamps[frame_ind2.cpu()]
        
        if self.opt_track:
            rots1 = self.input_rots[frame_ind1, column_ind1]
            rots2 = self.input_rots[frame_ind2, column_ind2]
            opt_rots1 = torch.zeros_like(rots1)
            opt_rots2 = torch.zeros_like(rots2)
            opt_rots1[0] = torch.cos(self.opt_rots[frame_ind1, column_ind1])
            opt_rots1[3] = torch.sin(self.opt_rots[frame_ind1, column_ind1])
            opt_rots2[0] = torch.cos(self.opt_rots[frame_ind2, column_ind2])
            opt_rots2[3] = torch.sin(self.opt_rots[frame_ind2, column_ind2])            

            rots1 = quaternion_raw_multiply(rots1.unsqueeze(0), opt_rots1.unsqueeze(0)).squeeze(0)
            rots2 = quaternion_raw_multiply(rots2.unsqueeze(0), opt_rots2.unsqueeze(0)).squeeze(0)

        else:
            rots1 = self.input_rots[ind1]
            rots2 = self.input_rots[ind2]

        r = (timestamp - timestamp1) / (timestamp2 - timestamp1)
        rots = quaternion_slerp(rots1, rots2, r)

        return rots
   
    def get_tracking_rotation(self, track_id, camera: Camera):
        if self.opt_track and camera.meta['is_val']:
            timestamp1, timestamp2 = self.find_closest_camera_timestamps(track_id, camera)
            if timestamp1 is None:
                return self.get_tracking_rotation_(track_id, camera.meta['timestamp'])
            else:
                timestamp = camera.meta['timestamp']
                rots1 = self.get_tracking_rotation_(track_id, timestamp1)
                rots2 = self.get_tracking_rotation_(track_id, timestamp2)
                r = (timestamp - timestamp1) / (timestamp2 - timestamp1)
                rots = quaternion_slerp(rots1, rots2, r)
                return rots
        else:
            return self.get_tracking_rotation_(track_id, camera.meta['timestamp'])