import numpy as np
import os
import shutil
import sqlite3
import glob
import cv2
import sys
sys.path.append(os.getcwd())

from scipy.spatial.transform import Rotation as R
from lib.config import cfg
from lib.utils.kitti_utils import generate_dataparser_outputs
from lib.utils.data_utils import get_val_frames

def image_to_mask(datadir, image_filename):

    splits = image_filename.split('/')
    cam_id, scene_id, basename = splits[-3], splits[-2], splits[-1]
    if cam_id == 'image_02':
        mask_filename = os.path.join(datadir, 'moving_vehicle_bound_2', scene_id, basename)
    else:
        mask_filename = os.path.join(datadir, 'moving_vehicle_bound_3', scene_id, basename)
    return mask_filename

def run_colmap_kitti(result=None):
    data_path = cfg.source_path
    model_path = cfg.model_path
    scene_id = cfg.data.scene_id

    if result is None:
        result = generate_dataparser_outputs(
            datadir=data_path,
            scene_id=scene_id,
            selected_frames=cfg.data.selected_frames,        
        )

    # copy images
    colmap_dir = os.path.join(model_path, 'colmap')
    os.makedirs(colmap_dir, exist_ok=True)
    
    image_filenames = result['image_filenames']
    exts = result['exts']
    ixts = result['ixts']
    frames_idx = result['frames_idx']
    split_test = cfg.data.get('split_test', -1)
    split_train = cfg.data.get('split_train', -1)
    num_frames = len(image_filenames) // 2
    train_frames, test_frames = get_val_frames(
        num_frames, 
        test_every=split_test if split_test > 0 else None,
        train_every=split_train if split_train > 0 else None,
    )

    train_image_filenames = [image_filenames[i] for i, frame_idx in enumerate(frames_idx)
            if frame_idx in train_frames]
    
    test_image_filenames = [image_filenames[i] for i, frame_idx in enumerate(frames_idx)
            if frame_idx in test_frames]
    
    mask_image_filenames = [image_to_mask(data_path, image_filename) for image_filename in image_filenames]
    
    train_images_dir = os.path.join(colmap_dir, 'train_imgs')
    test_images_dir = os.path.join(colmap_dir, 'test_imgs')
    mask_images_dir = os.path.join(colmap_dir, 'mask')

    if os.path.exists(train_images_dir):
        os.system(f'rm -rf {train_images_dir}')
    os.makedirs(train_images_dir)
    if os.path.exists(test_images_dir):
        os.system(f'rm -rf {test_images_dir}')  
    os.makedirs(test_images_dir)
    if os.path.exists(mask_images_dir):
        os.system(f'rm -rf {mask_images_dir}')
    os.makedirs(mask_images_dir)

    # copy images
    for image_filename in train_image_filenames:
        splits = image_filename.split('/')
        cam_id, scene_id, basename = splits[-3], splits[-2], splits[-1]
        new_image_filename = '{}_{}'.format(cam_id, basename)
        shutil.copyfile(image_filename, os.path.join(train_images_dir, new_image_filename))
        
    for image_filename in test_image_filenames:
        splits = image_filename.split('/')
        cam_id, scene_id, basename = splits[-3], splits[-2], splits[-1]
        new_image_filename = '{}_{}'.format(cam_id, basename)
        shutil.copyfile(image_filename, os.path.join(test_images_dir, new_image_filename))    

    for image_filename in mask_image_filenames:
        splits = image_filename.split('/')
        cam_id, scene_id, basename = splits[-3], splits[-2], splits[-1]
        new_image_filename = '{}_{}'.format(cam_id, basename)
        new_mask_filename = f'{new_image_filename}.png'
        shutil.copyfile(image_filename, os.path.join(mask_images_dir, new_mask_filename))
    
    for image_filename in sorted(glob.glob(os.path.join(mask_images_dir, '*.png'))):
        mask = cv2.imread(image_filename)
        flip_mask = (255 - mask).astype(np.uint8)
        cv2.imwrite(image_filename, flip_mask)
    
    # load intrinsics
    ixt = ixts

    # https://colmap.github.io/faq.html#mask-image-regions
    os.system(f'colmap feature_extractor \
            --ImageReader.mask_path {mask_images_dir} \
            --ImageReader.camera_model SIMPLE_PINHOLE  \
            --ImageReader.single_camera 1 \
            --ImageReader.camera_params {ixt[0, 0]},{ixt[0, 2]},{ixt[1, 2]} \
            --database_path {colmap_dir}/database.db \
            --image_path {train_images_dir}')
    
    # load extrinsics   
    c2w_dict = dict()
    for i, image_filename in enumerate(image_filenames):
        splits = image_filename.split('/')
        cam_id, scene_id, basename = splits[-3], splits[-2], splits[-1]
        new_image_filename = '{}_{}'.format(cam_id, basename)
        c2w_dict[new_image_filename] = exts[i]

    sample_img = cv2.imread(image_filenames[0])
    img_h, img_w = sample_img.shape[:2]
    
    db_fn = f'{colmap_dir}/database.db'
    conn = sqlite3.connect(db_fn)
    c = conn.cursor()
    c.execute('SELECT * FROM images')
    result = c.fetchall()

    out_fn = f'{colmap_dir}/id_names.txt'
    with open(out_fn, 'w') as f:
        for i in result:
            f.write(str(i[0]) + ' ' + i[1] + '\n')
    f.close()

    path_idname = f'{colmap_dir}/id_names.txt'
    
    f_id_name= open(path_idname, 'r')
    f_id_name_lines= f_id_name.readlines()
    
    images_save_dir = f'{colmap_dir}/created/sparse/model'
    if not os.path.exists(images_save_dir):
        os.makedirs(images_save_dir)
        
    f_w = open(f'{colmap_dir}/created/sparse/model/images.txt', 'w')
    id_names = []
    for l in f_id_name_lines:
        l = l.strip().split(' ')
        id_ = int(l[0])
        name = l[1]
        id_names.append([id_, name])

    for i in range(len(id_names)):
        id_ = id_names[i][0]
        name = id_names[i][1]
        
        transform = c2w_dict[name]
        transform = np.linalg.inv(transform)

        r = R.from_matrix(transform[:3,:3])
        rquat= r.as_quat()  # The returned value is in scalar-last (x, y, z, w) format.
        rquat[0], rquat[1], rquat[2], rquat[3] = rquat[3], rquat[0], rquat[1], rquat[2]
        out = np.concatenate((rquat, transform[:3, 3]), axis=0)
        id_ = id_names[i][0]
        name = id_names[i][1]
        f_w.write(f'{id_} ')
        f_w.write(' '.join([str(a) for a in out.tolist()] ) )
        f_w.write(f' 1 {name}')
        f_w.write('\n\n')
    
    f_w.close()

    cameras_fn = os.path.join(images_save_dir, 'cameras.txt')
    with open(cameras_fn, 'w') as f:
        fx = ixt[0, 0]
        fy = ixt[1, 1]
        cx = ixt[0, 2]
        cy = ixt[1, 2]
        f.write(f'1 SIMPLE_PINHOLE {img_w} {img_h} {fx} {cx} {cy}')

    points3D_fn = os.path.join(images_save_dir, 'points3D.txt')
    os.system(f'touch {points3D_fn}')
    
    triangulated_dir = os.path.join(colmap_dir, 'triangulated/sparse/model')
    os.makedirs(triangulated_dir, exist_ok=True)
    os.system(f'colmap exhaustive_matcher \
            --database_path {colmap_dir}/database.db')
    
    os.system(f'colmap point_triangulator \
            --database_path {colmap_dir}/database.db \
            --image_path {train_images_dir} \
            --input_path {colmap_dir}/created/sparse/model --output_path {colmap_dir}/triangulated/sparse/model')

    os.system(f'rm -rf {train_images_dir}')
    os.system(f'rm -rf {test_images_dir}')  
    os.system(f'rm -rf {mask_images_dir}')
    
if __name__ == '__main__':
    run_colmap_kitti(result=None)