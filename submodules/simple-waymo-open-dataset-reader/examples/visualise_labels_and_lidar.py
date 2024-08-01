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
import cv2
import io
import sys

from simple_waymo_open_dataset_reader import WaymoDataFileReader
from simple_waymo_open_dataset_reader import dataset_pb2, label_pb2
from simple_waymo_open_dataset_reader import utils

import matplotlib.cm
cmap = matplotlib.cm.get_cmap("viridis")

def display_labels_on_image(camera_calibration, img, labels, visibility):

    # Get the transformation matrix from the vehicle frame to image space.
    vehicle_to_image = utils.get_image_transform(camera_calibration)

    # Draw all the groundtruth labels
    for label,vis in zip(labels, visibility):
        if vis:
            colour = (0,0,200)
        else:
            colour = (128,0,0)

        utils.draw_3d_box(img, vehicle_to_image, label, colour=colour)
    
def display_laser_on_image(img, pcl, vehicle_to_image):
    # Convert the pointcloud to homogeneous coordinates.
    pcl1 = np.concatenate((pcl,np.ones_like(pcl[:,0:1])),axis=1)

    # Transform the point cloud to image space.
    proj_pcl = np.einsum('ij,bj->bi', vehicle_to_image, pcl1) 

    # Filter LIDAR points which are behind the camera.
    mask = proj_pcl[:,2] > 0
    proj_pcl = proj_pcl[mask]
    proj_pcl_attr = pcl_attr[mask]

    # Project the point cloud onto the image.
    proj_pcl = proj_pcl[:,:2]/proj_pcl[:,2:3]

    # Filter points which are outside the image.
    mask = np.logical_and(
        np.logical_and(proj_pcl[:,0] > 0, proj_pcl[:,0] < img.shape[1]),
        np.logical_and(proj_pcl[:,1] > 0, proj_pcl[:,1] < img.shape[1]))

    proj_pcl = proj_pcl[mask]
    proj_pcl_attr = proj_pcl_attr[mask]

    # Colour code the points based on distance.
    coloured_intensity = 255*cmap(proj_pcl_attr[:,0]/30)

    # Draw a circle for each point.
    for i in range(proj_pcl.shape[0]):
        cv2.circle(img, (int(proj_pcl[i,0]),int(proj_pcl[i,1])), 1, coloured_intensity[i])

if len(sys.argv) != 2:
    print("""Usage: python display_laser_on_image.py <datafile>
Display the groundtruth 3D bounding boxes and LIDAR points on the front camera video stream.""")
    sys.exit(0)

# Open a .tfrecord
filename = sys.argv[1]
datafile = WaymoDataFileReader(filename)

# Generate a table of the offset of all frame records in the file.
table = datafile.get_record_table()

print("There are %d frames in this file." % len(table))


# Loop through the whole file
## and display 3D labels.
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
    
    # BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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

    # Count the points inside each label's box.
    counts = mask.sum(1)

    # Keep boxes which contain at least 10 LIDAR points.
    visibility = counts > 10

    # Display the LIDAR points on the image.
    display_laser_on_image(img, pcl, vehicle_to_image)

    # Display the label's 3D bounding box on the image.
    display_labels_on_image(camera_calibration, img, frame.laser_labels, visibility)

    # Display the image
    cv2.imshow("Image", img)
    cv2.waitKey(10)

