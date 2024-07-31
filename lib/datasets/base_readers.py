import numpy as np
from typing import NamedTuple
from lib.utils.graphics_utils import getWorld2View2, focal2fov, fov2focal, BasicPointCloud
from plyfile import PlyData, PlyElement

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    K: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    metadata: dict = dict()
    mask: np.array = None
    acc_mask: np.array = None
            
class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    metadata: dict = dict()
    novel_view_cameras: list = None

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {
        'translate': translate, 
        'radius': radius, 
        'center': center,
    }
    
def get_PCA_Norm(xyz):
    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(xyz)
    axis = pca.components_[-1]
    projected_point_cloud = np.dot(xyz, axis)
    min_value = np.min(projected_point_cloud)
    max_value = np.max(projected_point_cloud)

    radius = (max_value - min_value) / 2

    return {
        'radius': radius, 
    }

def get_Sphere_Norm(xyz):
    from lib.config import cfg
    xyz_max = np.max(xyz, axis=0)
    xyz_min = np.min(xyz, axis=0)
    center = (xyz_max + xyz_min) / 2
    radius = np.linalg.norm(xyz_max - xyz_min) / 2.
    scale = cfg.data.get('sphere_scale', 1.0)
    radius *= scale
    
    return {
        'radius': radius, 
        'center': center,
    }


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # set rgb to 0 - 255
    if rgb.max() <= 1. and rgb.min() >= 0:
        rgb = np.clip(rgb * 255, 0., 255.)
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)
