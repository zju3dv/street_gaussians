import os
import sys
import numpy as np
from PIL import Image
from lib.utils.graphics_utils import getWorld2View2, focal2fov, fov2focal, BasicPointCloud
from lib.utils.colmap_utils import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from lib.config import cfg
from lib.datasets.base_readers import CameraInfo, SceneInfo, getNerfppNorm, fetchPly, storePly

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        if intr.model == "SIMPLE_PINHOLE":
            focal_length = intr.params[0]
            cx = intr.params[1]
            cy = intr.params[2]
            FovY = focal2fov(focal_length, height)
            FovX = focal2fov(focal_length, width)
            K = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]]).astype(np.float32)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            cx = intr.params[2]
            cy = intr.params[3]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            K = np.array([[focal_length_x, 0, cx], [0, focal_length_y, cy], [0, 0, 1]]).astype(np.float32)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, K=K, 
            image=image, image_path=image_path, image_name=image_name,
            width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readColmapSceneInfo(path, images='images', split_test=8, **kwargs):
    colmap_basedir = os.path.join(path, 'sparse/0')
    if not os.path.exists(colmap_basedir):
        colmap_basedir = os.path.join(path, 'sparse')
    try:
        cameras_extrinsic_file = os.path.join(colmap_basedir, "images.bin")
        cameras_intrinsic_file = os.path.join(colmap_basedir, "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(colmap_basedir, "images.txt")
        cameras_intrinsic_file = os.path.join(colmap_basedir, "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if split_test == -1:
        train_cam_infos = cam_infos
        test_cam_infos = []
    else:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % split_test != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % split_test == 0]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(colmap_basedir, "points3D.ply")
    bin_path = os.path.join(colmap_basedir, "points3D.bin")
    txt_path = os.path.join(colmap_basedir, "points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info