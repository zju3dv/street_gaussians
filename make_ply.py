import numpy as np
import os
import shutil
from lib.config import cfg
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.models.gaussian_model import GaussianModel
from lib.datasets.dataset import Dataset
from lib.models.scene import Scene
from plyfile import PlyData, PlyElement


inverse_opacity = lambda x: np.log(x/(1-x))
inverse_scale = lambda x: np.log(x)

if __name__ == '__main__':
    frame_id = cfg.viewer.frame_id 
    
    dataset = Dataset()
    gaussians = StreetGaussianModel(dataset.scene_info.metadata)
    scene = Scene(gaussians=gaussians, dataset=dataset)
    train_cameras = scene.getTrainCameras()
    test_cameras = scene.getTestCameras()
    cameras = train_cameras + test_cameras
    cameras = list(sorted(cameras, key=lambda x: x.id))
    
    viewpoint_camera = None
    for camera in cameras:
        if camera.meta['frame_idx'] == frame_id:
            viewpoint_camera = camera
            break
        
    if viewpoint_camera is None:
        raise ValueError(f'Could not find camera with frame_idx {frame_id}')
    
    gaussians.set_visibility(list(set(gaussians.model_name_id.keys())))
    gaussians.parse_camera(camera=viewpoint_camera)

    xyz = gaussians.get_xyz.detach().cpu().numpy()    
    normals = np.zeros_like(xyz)
    
    f = gaussians.get_features.detach().transpose(1, 2).contiguous() # [n, 3, sh_degree]
    f_dc = f[..., :1].flatten(start_dim=1).cpu().numpy()
    f_rest = f[..., 1:].flatten(start_dim=1).cpu().numpy()
    opacities = gaussians.get_opacity.detach().cpu().numpy()
    opacities = np.clip(opacities, a_min=1e-6, a_max=1.-1e-6)
    opacities = inverse_opacity(opacities)
    
    scale = gaussians.get_scaling.detach().cpu().numpy()
    scale = inverse_scale(scale)
    
    rotation = gaussians.get_rotation.detach().cpu().numpy()

    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(f_dc.shape[1]):
        l.append('f_dc_{}'.format(i))
    for i in range(f_rest.shape[1]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(scale.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(rotation.shape[1]):
        l.append('rot_{}'.format(i))
    dtype_full = [(attribute, 'f4') for attribute in l]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    
    save_dir = os.path.join(cfg.model_path, 'viewer', f'{frame_id:06d}')
    pointcloud_dir = os.path.join(save_dir, 'point_cloud', f'iteration_{cfg.train.iterations}')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(pointcloud_dir, exist_ok=True)
    shutil.copyfile(os.path.join(cfg.model_path, 'cameras.json'), os.path.join(save_dir, 'cameras.json'))
    shutil.copyfile(os.path.join(cfg.model_path, 'cfg_args'), os.path.join(save_dir, 'cfg_args'))
    shutil.copyfile(os.path.join(cfg.model_path, 'input.ply'), os.path.join(save_dir, 'input.ply'))
    
    elements = PlyElement.describe(elements, 'vertex')
    PlyData([elements]).write(os.path.join(pointcloud_dir, 'point_cloud.ply'))
    