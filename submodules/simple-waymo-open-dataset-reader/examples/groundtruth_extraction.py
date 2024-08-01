# Copyright (c) 2019, Grégoire Payen de La Garanderie, Durham University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import math
import io
import sys

from simple_waymo_open_dataset_reader import WaymoDataFileReader
from simple_waymo_open_dataset_reader import dataset_pb2, label_pb2
from simple_waymo_open_dataset_reader import utils

if len(sys.argv) != 2:
    print("""Usage: python groundtruth_extraction.py <datafile>
Extract the groundtruth objects in the KITTI tracking benchmark format.""")
    sys.exit(0)

# Open a .tfrecord
filename = sys.argv[1]
datafile = WaymoDataFileReader(filename)

# Generate a table of the offset of all frame records in the file.
table = datafile.get_record_table()

# Dictionary mapping each object to a unique tracking ID.
object_ids = dict()

# Loop through the whole file
## and dump the label information.
for frameno,frame in enumerate(datafile):

    # Get the top laser information
    laser_name = dataset_pb2.LaserName.TOP
    laser = utils.get(frame.lasers, laser_name)
    laser_calibration = utils.get(frame.context.laser_calibrations, laser_name)

    # Parse the top laser range image and get the associated projection.
    ri, camera_projection, range_image_pose = utils.parse_range_image_and_camera_projection(laser)

    # Convert the range image to a point cloud.
    pcl, pcl_attr = utils.project_to_pointcloud(frame, ri, camera_projection, range_image_pose, laser_calibration)

    
    # Get the front camera information
    camera_name = dataset_pb2.CameraName.FRONT
    camera_calibration = utils.get(frame.context.camera_calibrations, camera_name)
    camera = utils.get(frame.images, camera_name)

    # Get the transformation matrix for the camera.
    vehicle_to_image = utils.get_image_transform(camera_calibration)

    # Decode the image
    img = utils.decode_image(camera)

    # Some of the labels might be fully hidden therefore we attempt to compute the label visibility
    # by counting the number of LIDAR points inside each label bounding box.

    # For each label, compute the transformation matrix from the vehicle space to the box space.
    vehicle_to_labels = [np.linalg.inv(utils.get_box_transformation_matrix(label.box)) for label in frame.laser_labels]
    vehicle_to_labels = np.stack(vehicle_to_labels)

    # Convert the pointcloud to homogeneous coordinates.
    pcl1 = np.concatenate((pcl,np.ones_like(pcl[:,0:1])),axis=1)

    # Transform the point cloud to the label space for each label.
    # proj_pcl shape is [label, LIDAR point, coordinates]
    proj_pcl = np.einsum('lij,bj->lbi', vehicle_to_labels, pcl1)

    # For each pair of LIDAR point & label, check if the point is inside the label's box.
    # mask shape is [label, LIDAR point]
    mask = np.logical_and.reduce(np.logical_and(proj_pcl >= -1, proj_pcl <= 1),axis=2)

    # Count the points inside each label's box
    counts = mask.sum(1)

    # Keep boxes which contain at least 10 LIDAR points.
    visibility = counts > 10

    # Get the transformation matrix from vehicle space to camera space.
    camera_extrinsic = np.array(camera_calibration.extrinsic.transform).reshape(4,4)
    camera_extrinsic_inv = np.linalg.inv(camera_extrinsic)

    # For each non occluded label
    for label,labelvis in zip(frame.laser_labels,visibility):
        if not labelvis:
            continue

        # Compute the corners of the 3D bounding box of each label
        # and check that the label is within the view frustum of the camera
        corners = utils.get_3d_box_projected_corners(vehicle_to_image, label)
        if corners is None:
            continue

        # Compute the 2D bounding box of the label
        bbox = utils.compute_2d_bounding_box(img.shape, corners)

        # Assign a category name to the label
        if label.type == label_pb2.Label.Type.TYPE_VEHICLE:
            name = "vehicle"
        elif label.type == label_pb2.Label.Type.TYPE_PEDESTRIAN:
            name = "pedestrian"
        elif label.type == label_pb2.Label.Type.TYPE_SIGN:
            name = "sign"
        elif label.type == label_pb2.Label.Type.TYPE_CYCLIST:
            name = "cyclist"
        else:
            continue

        # If this is a new object, create a tracking ID for it.
        if label.id not in object_ids:
            object_ids[label.id] = len(object_ids)

        # Transform the label position to the camera space.
        pos = (label.box.center_x,label.box.center_y,label.box.center_z,1)
        pos = np.matmul(camera_extrinsic_inv, pos)

        # Compute the relative angle
        alpha = label.box.heading - math.atan2(pos[2],pos[0])

        # Print the information in the KITTI tracking benchmark format.
        print(frameno, object_ids[label.id], name, 0, 0, alpha, *bbox, label.box.height, label.box.width, label.box.length, label.box.center_x, label.box.center_y, label.box.center_z, label.box.heading)

