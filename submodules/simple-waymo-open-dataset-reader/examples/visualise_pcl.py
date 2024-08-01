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

import open3d as o3d

import matplotlib.cm
cmap = matplotlib.cm.get_cmap("viridis")

if len(sys.argv) != 2:
    print("""Usage: python visual_pcl.py <datafile>
Visualise the LIDAR points.""")
    sys.exit(0)

# Open a .tfrecord
filename = sys.argv[1]
datafile = WaymoDataFileReader(filename)

# Generate a table of the offset of all frame records in the file.
table = datafile.get_record_table()

print("There are %d frames in this file." % len(table))

# Initialise the visualiser
vis = o3d.visualization.VisualizerWithKeyCallback()

vis.create_window()
pcd = o3d.geometry.PointCloud()

once = True

datafile_iter = iter(datafile)

def display_next_frame(event=None):
    global once

    frame = next(datafile_iter)

    # Get the top laser information
    laser_name = dataset_pb2.LaserName.TOP
    laser = utils.get(frame.lasers, laser_name)
    laser_calibration = utils.get(frame.context.laser_calibrations, laser_name)

    # Parse the top laser range image and get the associated projection.
    ri, camera_projection, range_image_pose = utils.parse_range_image_and_camera_projection(laser)

    # Convert the range image to a point cloud.
    pcl, pcl_attr = utils.project_to_pointcloud(frame, ri, camera_projection, range_image_pose, laser_calibration)

    # Update the geometry and render the pointcloud
    pcd.points = o3d.utility.Vector3dVector(pcl)
    if once:
        vis.add_geometry(pcd)
        once = False
    else:
        vis.update_geometry(pcd)

vis.register_key_callback(262, display_next_frame) # Right arrow key

display_next_frame()

while True:
    vis.poll_events()
    vis.update_renderer()

