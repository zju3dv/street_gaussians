import numpy as np
import torch
import copy
import torch.nn as nn
import cv2
import math
from PIL import Image
from tqdm import tqdm
from lib.utils.general_utils import PILtoTorch, NumpytoTorch, matrix_to_quaternion
from lib.utils.graphics_utils import fov2focal, getProjectionMatrix, getWorld2View2, getProjectionMatrixK
from lib.datasets.base_readers import CameraInfo
from lib.config import cfg
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

# if training, put everything to cuda
# image_to_cuda = (cfg.mode == 'train') 

class Camera(nn.Module):
    def __init__(
        self, 
        id,
        R, T, 
        FoVx, FoVy, K,
        image, image_name, 
        trans = np.array([0.0, 0.0, 0.0]), 
        scale = 1.0,
        metadata = dict(),
        masks = dict(),
    ):
        super(Camera, self).__init__()

        self.id = id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.K = K
        self.image_name = image_name
        self.trans, self.scale = trans, scale

        # meta and mask
        self.meta = metadata
        for name, mask in masks.items():
            setattr(self, name, mask)
        
        self.original_image = image.clamp(0, 1)                
        self.image_height, self.image_width = self.original_image.shape[1], self.original_image.shape[2]
        self.zfar = 1000.0
        self.znear = 0.001
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        
        if self.K is not None:
            self.projection_matrix = getProjectionMatrixK(znear=self.znear, zfar=self.zfar, K=self.K, H=self.image_height, W=self.image_width).transpose(0,1).cuda()
            self.K = torch.from_numpy(self.K).float().cuda()
        else:
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()

        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
        if 'ego_pose' in self.meta.keys():
            self.ego_pose = torch.from_numpy(self.meta['ego_pose']).float().cuda()
            del self.meta['ego_pose']
            
        if 'extrinsic' in self.meta.keys():
            self.extrinsic = torch.from_numpy(self.meta['extrinsic']).float().cuda()
            del self.meta['extrinsic']
                
    def set_extrinsic(self, c2w):
        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3].T
        T = w2c[:3, 3]
        
        # set R, T
        self.R = R
        self.T = T
        
        # change attributes associated with R, T
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, self.trans, self.scale)).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
    
    def set_intrinsic(self, K):
        self.K = torch.from_numpy(K).float().cuda()
        self.projection_matrix = getProjectionMatrixK(znear=self.znear, zfar=self.zfar, K=self.K, H=self.image_height, W=self.image_width).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
    
    def get_extrinsic(self):
        w2c = np.eye(4)
        w2c[:3, :3] = self.R.T
        w2c[:3, 3] = self.T
        c2w = np.linalg.inv(w2c)
        return c2w
    
    def get_intrinsic(self):
        ixt = self.K.cpu().numpy()
        return ixt
    
        
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

def loadmask(cam_info: CameraInfo, resolution, resize_mode):
    masks = dict()
    if cfg.data.type == 'Blender':
        resized_image_rgb = PILtoTorch(cam_info.image, resolution, resize_mode=Image.BILINEAR)
        assert resized_image_rgb.shape[1] == 4
        masks['original_mask'] = None
        masks['original_acc_mask'] = resized_image_rgb[3:4, ...].clamp(0, 1).bool()
    else:
        if cam_info.mask is not None:
            masks['original_mask'] = PILtoTorch(cam_info.mask, resolution, resize_mode=resize_mode).clamp(0, 1).bool()
        # else:
        #     masks['original_mask'] = None
            
        if cam_info.acc_mask is not None:
            masks['original_acc_mask'] = PILtoTorch(cam_info.acc_mask, resolution, resize_mode=resize_mode).clamp(0, 1).bool()
        # else:
        #     masks['original_acc_mask'] = None
                        
        if 'sky_mask' in cam_info.metadata:
            masks['original_sky_mask'] = PILtoTorch(cam_info.metadata['sky_mask'], resolution, resize_mode=resize_mode).clamp(0, 1).bool()
            del cam_info.metadata['sky_mask']
        # else:
        #     masks['original_sky_mask'] = None    
        
        if 'obj_bound' in cam_info.metadata:
            masks['original_obj_bound'] = PILtoTorch(cam_info.metadata['obj_bound'], resolution, resize_mode=resize_mode).clamp(0, 1).bool()
            del cam_info.metadata['obj_bound']
        
    return masks

def loadmetadata(metadata, resolution):
    output = copy.deepcopy(metadata)

    

    # semantic
    if 'semantic' in metadata:
        output['semantic'] = NumpytoTorch(metadata['semantic'], resolution, resize_mode=Image.NEAREST)
    
    # lidar_depth
    if 'lidar_depth' in metadata:
        output['lidar_depth'] = NumpytoTorch(metadata['lidar_depth'], resolution, resize_mode=Image.NEAREST)
    
    # mono depth
    if 'mono_depth' in metadata:
        output['mono_depth'] = NumpytoTorch(metadata['mono_depth'], resolution, resize_mode=Image.NEAREST)
        
    # mono normal
    if 'mono_normal' in metadata:
        output['mono_normal'] = NumpytoTorch(metadata['mono_normal'], resolution, resize_mode=Image.NEAREST)
    
    return output
        
WARNED = False
def loadCam(cam_info: CameraInfo, resolution_scale):
    orig_w, orig_h = cam_info.image.size
    if cfg.resolution in [1, 2, 4, 8]:
        scale = resolution_scale * cfg.resolution
        resolution = round(orig_w / scale), round(orig_h / scale)
    else:  # should be a type that converts to float
        if cfg.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / cfg.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    K = copy.deepcopy(cam_info.K)
    K[:2] /= scale

    image = PILtoTorch(cam_info.image, resolution, resize_mode=Image.BILINEAR)[:3, ...]
    masks = loadmask(cam_info, resolution, resize_mode=Image.NEAREST)
    metadata = loadmetadata(cam_info.metadata, resolution)
    
    return Camera(
        id=cam_info.uid, 
        R=cam_info.R, 
        T=cam_info.T, 
        FoVx=cam_info.FovX, 
        FoVy=cam_info.FovY, 
        K=K,
        image=image, 
        masks=masks,
        image_name=cam_info.image_name, 
        metadata=metadata,
    )

def cameraList_from_camInfos(cam_infos, resolution_scale):
    camera_list = []

    for i, cam_info in tqdm(enumerate(cam_infos)):
        camera_list.append(loadCam(cam_info, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera: CameraInfo):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry

def make_rasterizer(
    viewpoint_camera: Camera,
    active_sh_degree = 0,
    bg_color = None,
    scaling_modifier = None,
):
    if bg_color is None:
        bg_color = [1, 1, 1] if cfg.data.white_background else [0, 0, 0]
        bg_color = torch.tensor(bg_color).float().cuda()
    if scaling_modifier is None:
        scaling_modifier = cfg.render.scaling_modifier
    debug = cfg.render.debug
    
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=debug,
    )    
            
    rasterizer: GaussianRasterizer = GaussianRasterizer(raster_settings=raster_settings)
    return rasterizer
