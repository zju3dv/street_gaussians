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
import cv2
import io
import sys

from simple_waymo_open_dataset_reader import WaymoDataFileReader
from simple_waymo_open_dataset_reader import dataset_pb2
from simple_waymo_open_dataset_reader import utils

def display_labels_on_image(camera_calibration, camera, labels, display_time = -1):
    # Get the image transformation matrix
    vehicle_to_image = utils.get_image_transform(camera_calibration)

    # Decode the JPEG image
    img = utils.decode_image(camera)

    # Draw all the groundtruth labels
    for label in labels:
        utils.draw_3d_box(img, vehicle_to_image, label)

    # Display the image
    cv2.imshow("Image", img)
    cv2.waitKey(display_time)
    
if len(sys.argv) != 2:
    print("""Usage: python visualise_labels.py <datafile>
Display the groundtruth 3D bounding boxes on the front camera video stream.""")
    sys.exit(0)

# Open a .tfrecord
filename = sys.argv[1]
datafile = WaymoDataFileReader(filename)

# Generate a table of the offset of all frame records in the file.
table = datafile.get_record_table()

print("There are %d frames in this file." % len(table))

# Loop through the whole file
## and display 3D labels.
for frame in datafile:
    camera_name = dataset_pb2.CameraName.FRONT
    camera_calibration = utils.get(frame.context.camera_calibrations, camera_name)
    camera = utils.get(frame.images, camera_name)

    display_labels_on_image(camera_calibration, camera, frame.laser_labels, 10)

# Alternative: Displaying a single frame:
# # Jump to the frame 150
# datafile.seek(table[150])
# 
# # Read and display this frame
# frame = datafile.read_record()
# display_labels_on_image(frame.context.camera_calibrations[0], frame.images[0], frame.laser_labels)

# Alternative: Displaying a 10 frames:
# # Jump to the frame 150
# datafile.seek(table[150])
# 
# for _ in range(10):
#     # Read and display this frame
#     frame = datafile.read_record()
#     display_labels_on_image(frame.context.camera_calibrations[0], frame.images[0], frame.laser_labels, 10)


