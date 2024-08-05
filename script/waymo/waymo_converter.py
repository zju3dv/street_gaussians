import sys
import os
sys.path.append(os.getcwd())
import torch
import numpy as np
import cv2
import math
import imageio
import argparse
import json
from tqdm import tqdm
from simple_waymo_open_dataset_reader import WaymoDataFileReader
from simple_waymo_open_dataset_reader import dataset_pb2, label_pb2
from simple_waymo_open_dataset_reader import utils
from lib.utils.box_utils import bbox_to_corner3d, get_bound_2d_mask
from lib.utils.img_utils import draw_3d_box_on_img
from lib.utils.graphics_utils import project_numpy

# castrack_path = '/nas/home/yanyunzhi/waymo/castrack/seq_infos/val/result.json'
# with open(castrack_path, 'r') as f:
#     castrack_infos = json.load(f)

camera_names_dict = {
    dataset_pb2.CameraName.FRONT_LEFT: 'FRONT_LEFT', 
    dataset_pb2.CameraName.FRONT_RIGHT: 'FRONT_RIGHT',
    dataset_pb2.CameraName.FRONT: 'FRONT', 
    dataset_pb2.CameraName.SIDE_LEFT: 'SIDE_LEFT',
    dataset_pb2.CameraName.SIDE_RIGHT: 'SIDE_RIGHT',
}

image_heights = [1280, 1280, 1280, 886, 886]
image_widths = [1920, 1920, 1920, 1920, 1920]

laser_names_dict = {
    dataset_pb2.LaserName.TOP: 'TOP',
    dataset_pb2.LaserName.FRONT: 'FRONT',
    dataset_pb2.LaserName.SIDE_LEFT: 'SIDE_LEFT',
    dataset_pb2.LaserName.SIDE_RIGHT: 'SIDE_RIGHT',
    dataset_pb2.LaserName.REAR: 'REAR',
}

opencv2camera = np.array([[0., 0., 1., 0.],
                        [-1., 0., 0., 0.],
                        [0., -1., 0., 0.],
                        [0., 0., 0., 1.]])

def get_extrinsic(camera_calibration):
    camera_extrinsic = np.array(camera_calibration.extrinsic.transform).reshape(4, 4) # camera to vehicle
    extrinsic = np.matmul(camera_extrinsic, opencv2camera) # [forward, left, up] to [right, down, forward]
    return extrinsic
    
def get_intrinsic(camera_calibration):
    camera_intrinsic = camera_calibration.intrinsic
    fx = camera_intrinsic[0]
    fy = camera_intrinsic[1]
    cx = camera_intrinsic[2]
    cy = camera_intrinsic[3]
    intrinsic = np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]])
    return intrinsic

def project_label_to_image(dim, obj_pose, calibration):
    bbox_l, bbox_w, bbox_h = dim
    bbox = np.array([[-bbox_l, -bbox_w, -bbox_h], [bbox_l, bbox_w, bbox_h]]) * 0.5
    points = bbox_to_corner3d(bbox)
    points = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)
    points_vehicle = points @ obj_pose.T # 3D bounding box in vehicle frame
    extrinsic = get_extrinsic(calibration)
    intrinsic = get_intrinsic(calibration)
    width, height = calibration.width, calibration.height
    points_uv, valid = project_numpy(
        xyz=points_vehicle[..., :3], 
        K=intrinsic, 
        RT=np.linalg.inv(extrinsic), 
        H=height, W=width
    )
    return points_uv, valid

def project_label_to_mask(dim, obj_pose, calibration):
    bbox_l, bbox_w, bbox_h = dim
    bbox = np.array([[-bbox_l, -bbox_w, -bbox_h], [bbox_l, bbox_w, bbox_h]]) * 0.5
    points = bbox_to_corner3d(bbox)
    points = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)
    points_vehicle = points @ obj_pose.T # 3D bounding box in vehicle frame
    extrinsic = get_extrinsic(calibration)
    intrinsic = get_intrinsic(calibration)
    width, height = calibration.width, calibration.height
    mask = get_bound_2d_mask(
        corners_3d=points_vehicle[..., :3],
        K=intrinsic,
        pose=np.linalg.inv(extrinsic), 
        H=height, W=width
    )
    
    return mask
    
    
def parse_seq_rawdata(process_list, root_dir, seq_name, seq_save_dir, track_file, start_idx=None, end_idx=None):
    print(f'Processing sequence {seq_name}...')
    print(f'Saving to {seq_save_dir}')
    
    try:
        with open(track_file, 'r') as f:
            castrack_infos = json.load(f)
    except:
        castrack_infos = dict()

    os.makedirs(seq_save_dir, exist_ok=True)
    
    seq_path = os.path.join(root_dir, seq_name+'.tfrecord')
    
    # set start and end timestep
    datafile = WaymoDataFileReader(seq_path)
    num_frames = len(datafile.get_record_table())
    start_idx = start_idx or 0
    end_idx = end_idx or num_frames - 1
    
    if 'pose' in process_list:
        ego_pose_save_dir = os.path.join(seq_save_dir, 'ego_pose')
        os.makedirs(ego_pose_save_dir, exist_ok=True)
        print("Processing ego pose...")
        timestamp = dict()
        timestamp['FRAME'] = dict()
        for camera_name in camera_names_dict.values():
            timestamp[camera_name] = dict()
        
        datafile = WaymoDataFileReader(seq_path)
        for frame_id, frame in tqdm(enumerate(datafile)):
            pose = np.array(frame.pose.transform).reshape(4, 4)
            np.savetxt(os.path.join(ego_pose_save_dir, f"{str(frame_id).zfill(6)}.txt"), pose)
            timestamp['FRAME'][str(frame_id).zfill(6)] = frame.timestamp_micros / 1e6
            
            camera_calibrations = frame.context.camera_calibrations
            for i, camera in enumerate(camera_calibrations):
                camera_name = camera.name
                camera_name_str = camera_names_dict[camera_name]
                camera = utils.get(frame.images, camera_name)
                camera_timestamp = camera.pose_timestamp
                timestamp[camera_name_str][str(frame_id).zfill(6)] = camera_timestamp
                
                camera_pose = np.array(camera.pose.transform).reshape(4, 4)
                np.savetxt(os.path.join(ego_pose_save_dir, f"{str(frame_id).zfill(6)}_{camera_name-1}.txt"), camera_pose)

        timestamp_save_path = os.path.join(seq_save_dir, "timestamps.json")
        with open(timestamp_save_path, 'w') as f:
            json.dump(timestamp, f, indent=1)

    
    if 'calib' in process_list:
        intrinsic_save_dir = os.path.join(seq_save_dir, 'intrinsics')
        extrinsic_save_dir = os.path.join(seq_save_dir, 'extrinsics')
        os.makedirs(intrinsic_save_dir, exist_ok=True)
        os.makedirs(extrinsic_save_dir, exist_ok=True)
        print("Processing camera calibration...")
        
        datafile = WaymoDataFileReader(seq_path)
        for frame_id, frame in tqdm(enumerate(datafile)):
            camera_calibrations = frame.context.camera_calibrations
        
        extrinsics = []
        intrinsics = []
        camera_names = []
        for camera in camera_calibrations:
            extrinsic = np.array(camera.extrinsic.transform).reshape(4, 4)
            extrinsic = np.matmul(extrinsic, opencv2camera) # [forward, left, up] to [right, down, forward]
            intrinsic = list(camera.intrinsic)
            extrinsics.append(extrinsic)
            intrinsics.append(intrinsic)
            camera_names.append(camera.name)
        
        for i in range(5):
            np.savetxt(os.path.join(extrinsic_save_dir, f"{str(camera_names[i] - 1)}.txt"), extrinsics[i])
            np.savetxt(os.path.join(intrinsic_save_dir, f"{str(camera_names[i] - 1)}.txt"), intrinsics[i])
    
    if 'image' in process_list:
        image_save_dir = os.path.join(seq_save_dir, 'images')
        os.makedirs(image_save_dir, exist_ok=True)        
        print("Processing image data...")
        
        datafile = WaymoDataFileReader(seq_path)
        for frame_id, frame in tqdm(enumerate(datafile)):
            for camera_name, camera_name_str in camera_names_dict.items():                    
                camera = utils.get(frame.images, camera_name)
                img = utils.decode_image(camera)
                img_path = os.path.join(image_save_dir, f'{frame_id:06d}_{str(camera.name - 1)}.png')
                cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        print("Processing image data done...")

    if 'lidar' in process_list:
        pts_3d_all = dict()
        pts_2d_all = dict()
        print("Processing LiDAR data...")
                
        datafile = WaymoDataFileReader(seq_path)
        for frame_id, frame in tqdm(enumerate(datafile)):
            pts_3d = [] # LiDAR point cloud in world frame
            pts_2d = [] # LiDAR point cloud projection in camera [camera_name, w, h] 
            
            for laser_name, laser_name_str in laser_names_dict.items():
                laser = utils.get(frame.lasers, laser_name)
                laser_calibration = utils.get(frame.context.laser_calibrations, laser_name)
                ri, camera_projection, range_image_pose = utils.parse_range_image_and_camera_projection(laser)

                # LiDAR spherical coordinate -> polar -> cartesian
                pcl, pcl_attr = utils.project_to_pointcloud(frame, ri, camera_projection, range_image_pose, laser_calibration)
                pts_3d.append(pcl[:, :3]) # save LiDAR pointcloud in vehicle frame 
                
                # Transform LIDAR point cloud from vehicle frame to world frame
                # vehicle_pose = np.array(frame.pose.transform).reshape(4, 4)
                # pcl = vehicle_pose.dot(np.concatenate([pcl, np.ones((pcl.shape[0], 1))], axis=1).T).T
                                    
                mask = ri[:, :, 0] > 0
                camera_projection = camera_projection[mask]

                # Can be projected to multi-cameras, order: [FRONT, FRONT_LEFT, FRONT_RIGHT, SIDE_LEFT, SIDE_RIGHT].
                # Only save the first projection camera,

                # camera_projection
                # Inner dimensions are:
                # channel 0: CameraName.Name of 1st projection. Set to UNKNOWN if no projection.
                # channel 1: x (axis along image width)
                # channel 2: y (axis along image height)
                # channel 3: CameraName.Name of 2nd projection. Set to UNKNOWN if no projection.
                # channel 4: x (axis along image width)
                # channel 5: y (axis along image height)
                # Note: pixel 0 corresponds to the left edge of the first pixel in the image.
                
                camera_projection[:, 0] -= 1
                camera_projection[:, 3] -= 1
                camera_projection = camera_projection.astype(np.int16)
                
                pts_2d.append(camera_projection)

            pts_3d = np.concatenate(pts_3d, axis=0)
            pts_3d_all[frame_id] = pts_3d
            pts_2d = np.concatenate(pts_2d, axis=0)
            pts_2d_all[frame_id] = pts_2d
                                                
        np.savez_compressed(f'{seq_save_dir}/pointcloud.npz', 
                            pointcloud=pts_3d_all, 
                            camera_projection=pts_2d_all)
        print("Processing LiDAR data done...")

    if 'track' in process_list:
        print("Processing tracking data...")
        track_dir = os.path.join(seq_save_dir, "track")
        os.makedirs(track_dir, exist_ok=True)
        
        # Use GT tracker
        track_infos_path = os.path.join(track_dir, "track_info.txt")
        track_infos_file = open(track_infos_path, 'w')
        row_info_title = "frame_id " + "track_id " + "object_class " + "alpha " + \
                "box_height " + "box_width " + "box_length " + "box_center_x " + "box_center_y " + "box_center_z " \
                + "box_heading " \
                + "speed" + "\n"

        track_infos_file.write(row_info_title)
        track_vis_imgs = []
        bbox_visible_dict = dict()
        object_ids = dict()

        datafile = WaymoDataFileReader(seq_path)

        for frame_id, frame in tqdm(enumerate(datafile)):
            images = dict()
            for camera_name in camera_names_dict.keys():
                camera = utils.get(frame.images, camera_name)
                image = utils.decode_image(camera)
                images[camera_name] = image
            
            for label in frame.laser_labels:
                box = label.box
                
                # build 3D bounding box dimension
                length, width, height = box.length, box.width, box.height
                
                # build 3D bounding box pose
                tx, ty, tz = box.center_x, box.center_y, box.center_z
                heading = box.heading
                c = math.cos(heading)
                s = math.sin(heading)
                rotz_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

                obj_pose_vehicle = np.eye(4)
                obj_pose_vehicle[:3, :3] = rotz_matrix
                obj_pose_vehicle[:3, 3] = np.array([tx, ty, tz])

                if label.id not in object_ids:
                    object_ids[label.id] = len(object_ids)

                label_id = object_ids[label.id]
                if label_id not in bbox_visible_dict:
                    bbox_visible_dict[label_id] = dict()
                bbox_visible_dict[label_id][frame_id] = []

                for camera_name in camera_names_dict.keys():
                    camera_calibration = utils.get(frame.context.camera_calibrations, camera_name)
                
                    vertices, valid = project_label_to_image(
                        dim=[length, width, height],
                        obj_pose=obj_pose_vehicle,
                        calibration=camera_calibration,
                    )
                    
                    # if one corner of the 3D bounding box is on camera plane, we should consider it as visible
                    # partial visible for the case when not all corners can be observed
                    if valid.any():
                        # print(f'At frame {frame_id}, label {label_id} is visible on {camera_names_dict[camera_name]}')
                        bbox_visible_dict[label_id][frame_id].append(camera_name-1)
                    if valid.all():
                        vertices = vertices.reshape(2, 2, 2, 2).astype(np.int32)
                        draw_3d_box_on_img(vertices, images[camera_name])
                    
                bbox_visible_dict[label_id][frame_id] = sorted(bbox_visible_dict[label_id][frame_id])
                    
                # assume every bbox is visible in at least on camera
                if label.type == label_pb2.Label.Type.TYPE_VEHICLE:
                    obj_class = "vehicle"
                elif label.type == label_pb2.Label.Type.TYPE_PEDESTRIAN:
                    obj_class = "pedestrian"
                elif label.type == label_pb2.Label.Type.TYPE_SIGN:
                    obj_class = "sign"
                elif label.type == label_pb2.Label.Type.TYPE_CYCLIST:
                    obj_class = "cyclist"
                else:
                    obj_class = "misc"

                alpha = -10
                meta = label.metadata
                speed = np.linalg.norm([meta.speed_x, meta.speed_y])  
                lines_info = f"{frame_id} {label_id} {obj_class} {alpha} {height} {width} {length} {tx} {ty} {tz} {heading} {speed} \n"
                
                track_infos_file.write(lines_info)
                
            track_vis_img = np.concatenate([
                images[dataset_pb2.CameraName.FRONT_LEFT], 
                images[dataset_pb2.CameraName.FRONT], 
                images[dataset_pb2.CameraName.FRONT_RIGHT]], axis=1)
            track_vis_imgs.append(track_vis_img)
        
        # save track visualization        
        imageio.mimwrite(os.path.join(track_dir, "track_vis.mp4"), track_vis_imgs, fps=24)
        
        # save bbox visibility
        bbox_visible_path = os.path.join(track_dir, "track_camera_vis.json")
        with open(bbox_visible_path, 'w') as f:
            json.dump(bbox_visible_dict, f, indent=1)
            
        # save object ids mapping
        object_ids_path = os.path.join(track_dir, "track_ids.json")
        with open(object_ids_path, 'w') as f:
            json.dump(object_ids, f, indent=2)
        
        track_infos_file.close()

        # Use castrack
        if seq_name in castrack_infos:     
            track_infos_path = os.path.join(track_dir, "track_info_castrack.txt")
            track_infos_file = open(track_infos_path, 'w')
            row_info_title = "frame_id " + "track_id " + "object_class " + "alpha " + \
                "box_height " + "box_width " + "box_length " + "box_center_x " + "box_center_y " + "box_center_z " \
                + "box_heading " + "\n"
            track_infos_file.write(row_info_title)

            track_info = castrack_infos[seq_name]
            track_vis_imgs = []
            bbox_visible_dict = dict()
            object_ids = dict()

            datafile = WaymoDataFileReader(seq_path)
            for frame_id, frame in tqdm(enumerate(datafile)):
                images = dict()
                for camera_name in camera_names_dict.keys():
                    camera = utils.get(frame.images, camera_name)
                    image = utils.decode_image(camera)
                    images[camera_name] = image
                
                label = track_info[str(frame_id)]
                for i, object_id in enumerate(label['obj_ids']):
                    box = label['boxes_lidar'][i]

                    # build 3D bounding box dimension
                    length, width, height = box[3], box[4], box[5]
                    
                    # build 3D bounding box pose
                    tx, ty, tz = box[0], box[1], box[2]
                    heading = box[-1]
                    c = math.cos(heading)
                    s = math.sin(heading)
                    rotz_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                    obj_pose_vehicle = np.eye(4)
                    obj_pose_vehicle[:3, :3] = rotz_matrix
                    obj_pose_vehicle[:3, 3] = np.array([tx, ty, tz])
                    
                    if object_id not in object_ids:
                        object_ids[object_id] = len(object_ids)
                        
                    label_id = object_ids[object_id]
                    if label_id not in bbox_visible_dict:
                        bbox_visible_dict[label_id] = dict()
                    bbox_visible_dict[label_id][frame_id] = []
                    
                    for camera_name in camera_names_dict.keys():
                        camera_calibration = utils.get(frame.context.camera_calibrations, camera_name)
                        
                        vertices, valid = project_label_to_image(
                            dim=[length, width, height],
                            obj_pose=obj_pose_vehicle,
                            calibration=camera_calibration,
                        )
                        
                        # if one corner of the 3D bounding box is on camera plane, we should consider it as visible
                        # partial visible for the case when not all corners can be observed
                        if valid.any():
                            # print(f'At frame {frame_id}, label {label_id} is visible on {camera_names_dict[camera_name]}')
                            bbox_visible_dict[label_id][frame_id].append(camera_name-1)
                        if valid.all():
                            vertices = vertices.reshape(2, 2, 2, 2).astype(np.int32)
                            draw_3d_box_on_img(vertices, images[camera_name])
                      
                    bbox_visible_dict[label_id][frame_id] = sorted(bbox_visible_dict[label_id][frame_id])

                    # assume every bbox is visible in at least one camera
                    name = label['name'][i]
                    if name == 'Vehicle':
                        obj_class = "vehicle"
                    elif name == 'Cyclist':
                        obj_class = "cyclist"
                    elif name == 'Pedestrian':
                        obj_class = "pedestrian"
                    else:
                        obj_class = "misc"

                    alpha = -10

                    lines_info = f"{frame_id} {label_id} {obj_class} {alpha} {height} {width} {length} {tx} {ty} {tz} {heading} \n"
                    
                    track_infos_file.write(lines_info)
                
                track_vis_img = np.concatenate([
                    images[dataset_pb2.CameraName.FRONT_LEFT], 
                    images[dataset_pb2.CameraName.FRONT], 
                    images[dataset_pb2.CameraName.FRONT_RIGHT]], axis=1)
                track_vis_imgs.append(track_vis_img)
            
            # save track visualization
            imageio.mimwrite(os.path.join(track_dir, "track_vis_castrack.mp4"), track_vis_imgs, fps=24)

            # save bbox visibility
            bbox_visible_path = os.path.join(track_dir, "track_camera_vis_castrack.json")
            with open(bbox_visible_path, 'w') as f:
                json.dump(bbox_visible_dict, f, indent=1)

            # save object ids mapping
            object_ids_path = os.path.join(track_dir, "track_ids_castrack.json")
            with open(object_ids_path, 'w') as f:
                json.dump(object_ids, f, indent=2)

            track_infos_file.close()
            print("Processing tracking data done...")

    if 'dynamic_mask' in process_list:
        print("Saving dynamic mask ...")
        dynamic_mask_dir = os.path.join(seq_save_dir, "dynamic_mask")
        os.makedirs(dynamic_mask_dir, exist_ok=True)
        datafile = WaymoDataFileReader(seq_path)

        for frame_id, frame in tqdm(enumerate(datafile)):
            masks = dict()
            for camera_name in camera_names_dict.keys():
                camera_calibration = utils.get(frame.context.camera_calibrations, camera_name)
                width, height = camera_calibration.width, camera_calibration.height
                mask = np.zeros((height, width), dtype=np.uint8)
                masks[camera_name] = mask
    
            for label in frame.laser_labels:
                box = label.box
                meta = label.metadata
                speed = np.linalg.norm([meta.speed_x, meta.speed_y]) 
                
                # thresholding, use 1.0 m/s to determine whether the pixel is moving
                # follow EmerNeRF
                if speed < 1.:
                    continue
                
                # build 3D bounding box dimension
                length, width, height = box.length, box.width, box.height
                
                # build 3D bounding box pose
                tx, ty, tz = box.center_x, box.center_y, box.center_z
                heading = box.heading
                c = math.cos(heading)
                s = math.sin(heading)
                rotz_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

                obj_pose_vehicle = np.eye(4)
                obj_pose_vehicle[:3, :3] = rotz_matrix
                obj_pose_vehicle[:3, 3] = np.array([tx, ty, tz])

                for camera_name in camera_names_dict.keys():
                    camera_calibration = utils.get(frame.context.camera_calibrations, camera_name)
                    # dim = [length * 1.5, width * 1.5, height]
                    dim = [length, width, height]
                    vertices, valid = project_label_to_image(
                        dim=dim,
                        obj_pose=obj_pose_vehicle,
                        calibration=camera_calibration,
                    )
                    if valid.any():
                        mask = project_label_to_mask(
                            dim=dim,
                            obj_pose=obj_pose_vehicle,
                            calibration=camera_calibration,
                        )
                        masks[camera_name] = np.logical_or(
                            masks[camera_name], mask)
            
            for camera_name in camera_names_dict.keys():
                mask = masks[camera_name]
                mask_path = os.path.join(dynamic_mask_dir, f'{frame_id:06d}_{str(camera_name - 1)}.png')
                cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))

        print("Saving dynamic mask done...")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--process_list', type=str, nargs='+', default=['pose', 'calib', 'image', 'lidar', 'track', 'dynamic_mask'])
    parser.add_argument('--root_dir', type=str, default='/nas/home/yanyunzhi/waymo/training')
    parser.add_argument('--save_dir', type=str, default='/nas/home/yanyunzhi/waymo/street_gaussian/training/surrounding')
    parser.add_argument('--track_file', type=str, default='/nas/home/yanyunzhi/waymo/castrack/seq_infos/val/result.json')
    parser.add_argument('--split_file', type=str)
    parser.add_argument('--segment_file', type=str)
    args = parser.parse_args()
    
    process_list = args.process_list
    root_dir = args.root_dir
    save_dir = args.save_dir
    track_file = args.track_file
    split_file = open(args.split_file, "r").readlines()[1:]
    scene_ids_list = [int(line.strip().split(",")[0]) for line in split_file]
    seq_names = [line.strip().split(",")[1] for line in split_file]
    segment_file = args.segment_file

    seq_lists = open(segment_file).read().splitlines()
    # seq_lists = open(os.path.join(root_dir, 'segment_list.txt')).read().splitlines()
    os.makedirs(save_dir, exist_ok=True)
    for i, scene_id in enumerate(scene_ids_list):
        assert seq_names[i][3:] == seq_lists[scene_id][8:14]
        seq_save_dir = os.path.join(save_dir, str(scene_id).zfill(3))
        parse_seq_rawdata(
            process_list=process_list,
            root_dir=root_dir,
            seq_name=seq_lists[scene_id],
            seq_save_dir=seq_save_dir,
            track_file=track_file,
        )
    
if __name__ == '__main__':
    main()