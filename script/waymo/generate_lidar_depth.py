import sys
import os
sys.path.append(os.getcwd())
import argparse
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from lib.utils.img_utils import visualize_depth_numpy

image_filename_to_cam = lambda x: int(x.split('.')[0][-1])
image_filename_to_frame = lambda x: int(x.split('.')[0][:6])
def load_calibration(datadir):
    extrinsics_dir = os.path.join(datadir, 'extrinsics')
    intrinsics_dir = os.path.join(datadir, 'intrinsics')
    
    intrinsics = []
    extrinsics = []
    for i in range(5):
        intrinsic = np.loadtxt(os.path.join(intrinsics_dir,  f"{i}.txt"))
        fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        intrinsics.append(intrinsic)
        cam_to_ego = np.loadtxt(os.path.join(extrinsics_dir,  f"{i}.txt"))
    
    for i in range(5):
        cam_to_ego = np.loadtxt(os.path.join(extrinsics_dir,  f"{i}.txt"))
        extrinsics.append(cam_to_ego)
        
    return extrinsics, intrinsics

# single frame sparse lidar depth
def generate_lidar_depth(datadir):
    save_dir = os.path.join(datadir, 'lidar_depth')
    os.makedirs(save_dir, exist_ok=True)
    
    image_dir = os.path.join(datadir, 'images')
    image_files = glob(image_dir + "/*.jpg") 
    image_files += glob(image_dir + "/*.png")
    image_files = sorted(image_files)
    
    
    pointcloud_path = os.path.join(datadir, 'pointcloud.npz')
    pts3d_dict = np.load(pointcloud_path, allow_pickle=True)['pointcloud'].item()
    pts2d_dict = np.load(pointcloud_path, allow_pickle=True)['camera_projection'].item()  
    
    extrinsics, intrinsics = load_calibration(datadir)
    for image_filename in tqdm(image_files):
        image = cv2.imread(image_filename)
        h, w = image.shape[:2]
        
        image_basename = os.path.basename(image_filename)
        frame = image_filename_to_frame(image_basename)
        cam = image_filename_to_cam(image_basename)
        
        depth_path = os.path.join(save_dir, f'{os.path.basename(image_filename).split(".")[0]}.npy')
        depth_vis_path = os.path.join(save_dir, f'{os.path.basename(image_filename).split(".")[0]}.png')
        
        raw_3d = pts3d_dict[frame]
        raw_2d = pts2d_dict[frame]
            
        num_pts = raw_3d.shape[0]
        pts_idx = np.arange(num_pts)
        pts_idx = np.tile(pts_idx[..., None], (1, 2)).reshape(-1) # (num_pts * 2)
        raw_2d = raw_2d.reshape(-1, 3) # (num_pts * 2, 3)
        mask = (raw_2d[:, 0] == cam)
        
        points_xyz = raw_3d[pts_idx[mask]]
        points_xyz = np.concatenate([points_xyz, np.ones_like(points_xyz[..., :1])], axis=-1)

        c2w = extrinsics[cam]
        w2c = np.linalg.inv(c2w)
        points_xyz_cam = points_xyz @ w2c.T
        points_depth = points_xyz_cam[..., 2]

        valid_mask = points_depth > 0.
        
        points_xyz_pixel = raw_2d[mask][:, 1:3]
        points_coord = points_xyz_pixel[valid_mask].round().astype(np.int32)
        points_coord[:, 0] = np.clip(points_coord[:, 0], 0, w-1)
        points_coord[:, 1] = np.clip(points_coord[:, 1], 0, h-1)
        
        depth = (np.ones((h, w)) * np.finfo(np.float32).max).reshape(-1)
        u, v = points_coord[:, 0], points_coord[:, 1]
        indices = v * w + u
        np.minimum.at(depth, indices, points_depth[valid_mask])
        depth[depth >= np.finfo(np.float32).max - 1e-5] = 0
        valid_depth_pixel = (depth != 0)
        valid_depth_value = depth[valid_depth_pixel]
        valid_depth_pixel = valid_depth_pixel.reshape(h, w).astype(np.bool_)
                    
        depth_file = dict()
        depth_file['mask'] = valid_depth_pixel
        depth_file['value'] = valid_depth_value
        np.save(depth_path, depth_file)

        try:
            if cam == 0:
                depth = depth.reshape(h, w).astype(np.float32)
                depth_vis, _ = visualize_depth_numpy(depth)
                depth_on_img = image[..., [2, 1, 0]]
                depth_on_img[depth > 0] = depth_vis[depth > 0]
                cv2.imwrite(depth_vis_path, depth_on_img)      
        except:
            print(f'error in visualize depth of {image_filename}, depth range: {depth.min()} - {depth.max()}')
    

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', required=True, type=str)

    args = parser.parse_args()
    
    generate_lidar_depth(args.datadir)
    