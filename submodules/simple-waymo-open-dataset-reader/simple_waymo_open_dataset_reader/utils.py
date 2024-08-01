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
from simple_waymo_open_dataset_reader import dataset_pb2, label_pb2
import zlib
import math
import io


def get_box_transformation_matrix(box):
    """Create a transformation matrix for a given label box pose."""

    tx,ty,tz = box.center_x,box.center_y,box.center_z
    c = math.cos(box.heading)
    s = math.sin(box.heading)

    sl, sh, sw = box.length, box.height, box.width

    return np.array([
        [ sl*c,-sw*s,  0,tx],
        [ sl*s, sw*c,  0,ty],
        [    0,    0, sh,tz],
        [    0,    0,  0, 1]])

def get_3d_box_projected_corners(vehicle_to_image, label):
    """Get the 2D coordinates of the 8 corners of a label's 3D bounding box.

    vehicle_to_image: Transformation matrix from the vehicle frame to the image frame.
    label: The object label
    """

    box = label.box

    # Get the vehicle pose
    box_to_vehicle = get_box_transformation_matrix(box)

    # Calculate the projection from the box space to the image space.
    box_to_image = np.matmul(vehicle_to_image, box_to_vehicle)


    # Loop through the 8 corners constituting the 3D box
    # and project them onto the image
    vertices = np.empty([2,2,2,2])
    for k in [0, 1]:
        for l in [0, 1]:
            for m in [0, 1]:
                # 3D point in the box space
                v = np.array([(k-0.5), (l-0.5), (m-0.5), 1.])

                # Project the point onto the image
                v = np.matmul(box_to_image, v)

                # If any of the corner is behind the camera, ignore this object.
                if v[2] < 0:
                    return None

                vertices[k,l,m,:] = [v[0]/v[2], v[1]/v[2]]

    vertices = vertices.astype(np.int32)

    return vertices

def compute_2d_bounding_box(img_or_shape,points):
    """Compute the 2D bounding box for a set of 2D points.
    
    img_or_shape: Either an image or the shape of an image.
                  img_or_shape is used to clamp the bounding box coordinates.
    
    points: The set of 2D points to use
    """

    if isinstance(img_or_shape,tuple):
        shape = img_or_shape
    else:
        shape = img_or_shape.shape

    # Compute the 2D bounding box and draw a rectangle
    x1 = np.amin(points[...,0])
    x2 = np.amax(points[...,0])
    y1 = np.amin(points[...,1])
    y2 = np.amax(points[...,1])

    x1 = min(max(0,x1),shape[1])
    x2 = min(max(0,x2),shape[1])
    y1 = min(max(0,y1),shape[0])
    y2 = min(max(0,y2),shape[0])

    return (x1,y1,x2,y2)

def draw_3d_box(img, vehicle_to_image, label, colour=(255,128,128), draw_2d_bounding_box=False):
    """Draw a 3D bounding from a given 3D label on a given "img". "vehicle_to_image" must be a projection matrix from the vehicle reference frame to the image space.

    draw_2d_bounding_box: If set a 2D bounding box encompassing the 3D box will be drawn
    """
    import cv2

    vertices = get_3d_box_projected_corners(vehicle_to_image, label)

    if vertices is None:
        # The box is not visible in this image
        return

    if draw_2d_bounding_box:
        x1,y1,x2,y2 = compute_2d_bounding_box(img.shape, vertices)

        if (x1 != x2 and y1 != y2):
            cv2.rectangle(img, (x1,y1), (x2,y2), colour, thickness = 1)
    else:
        # Draw the edges of the 3D bounding box
        for k in [0, 1]:
            for l in [0, 1]:
                for idx1,idx2 in [((0,k,l),(1,k,l)), ((k,0,l),(k,1,l)), ((k,l,0),(k,l,1))]:
                    cv2.line(img, tuple(vertices[idx1]), tuple(vertices[idx2]), colour, thickness=1)
        # Draw a cross on the front face to identify front & back.
        for idx1,idx2 in [((1,0,0),(1,1,1)), ((1,1,0),(1,0,1))]:
            cv2.line(img, tuple(vertices[idx1]), tuple(vertices[idx2]), colour, thickness=1)

def draw_2d_box(img, label, colour=(255,128,128)):
    """Draw a 2D bounding from a given 2D label on a given "img".
    """
    import cv2

    box = label.box

    # Extract the 2D coordinates
    # It seems that "length" is the actual width and "width" is the actual height of the bounding box. Most peculiar.
    x1 = int(box.center_x - box.length/2)
    x2 = int(box.center_x + box.length/2)
    y1 = int(box.center_y - box.width/2)
    y2 = int(box.center_y + box.width/2)

    # Draw the rectangle
    cv2.rectangle(img, (x1,y1), (x2,y2), colour, thickness = 1)


def decode_image(camera):
    """ Decode the JPEG image. """

    from PIL import Image
    return np.array(Image.open(io.BytesIO(camera.image)))

def get_image_transform(camera_calibration):
    """ For a given camera calibration, compute the transformation matrix
        from the vehicle reference frame to the image space.
    """

    # TODO: Handle the camera distortions
    extrinsic = np.array(camera_calibration.extrinsic.transform).reshape(4,4)
    intrinsic = camera_calibration.intrinsic

    # Camera model:
    # | fx  0 cx 0 |
    # |  0 fy cy 0 |
    # |  0  0  1 0 |
    camera_model = np.array([
        [intrinsic[0], 0, intrinsic[2], 0],
        [0, intrinsic[1], intrinsic[3], 0],
        [0, 0,                       1, 0]])

    # Swap the axes around
    axes_transformation = np.array([
        [0,-1,0,0],
        [0,0,-1,0],
        [1,0,0,0],
        [0,0,0,1]])

    # Compute the projection matrix from the vehicle space to image space.
    vehicle_to_image = np.matmul(camera_model, np.matmul(axes_transformation, np.linalg.inv(extrinsic)))
    return vehicle_to_image

def get_rotation_matrix(roll, pitch, yaw):
    """ Convert Euler angles to a rotation matrix"""

    cos_roll = np.cos(roll)
    sin_roll = np.sin(roll)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    cos_pitch = np.cos(pitch)
    sin_pitch = np.sin(pitch)

    ones = np.ones_like(yaw)
    zeros = np.zeros_like(yaw)

    r_roll = np.stack([
        [ones,  zeros,     zeros],
        [zeros, cos_roll, -sin_roll],
        [zeros, sin_roll,  cos_roll]])

    r_pitch = np.stack([
        [ cos_pitch, zeros, sin_pitch],
        [ zeros,     ones,  zeros],
        [-sin_pitch, zeros, cos_pitch]])

    r_yaw = np.stack([
        [cos_yaw, -sin_yaw, zeros],
        [sin_yaw,  cos_yaw, zeros],
        [zeros,    zeros,   ones]])

    pose = np.einsum('ijhw,jkhw,klhw->ilhw',r_yaw,r_pitch,r_roll)
    pose = pose.transpose(2,3,0,1)
    return pose

def parse_range_image_and_camera_projection(laser, second_response=False):
    """ Parse the range image for a given laser.

    second_response: If true, return the second strongest response instead of the primary response.
                     The second_response might be useful to detect the edge of objects
    """

    range_image_pose = None
    camera_projection = None

    if not second_response:
        # Return the strongest response if available
        if len(laser.ri_return1.range_image_compressed) > 0:
            ri = dataset_pb2.MatrixFloat()
            ri.ParseFromString(
                zlib.decompress(laser.ri_return1.range_image_compressed))
            ri = np.array(ri.data).reshape(ri.shape.dims)

            if laser.name == dataset_pb2.LaserName.TOP:
                range_image_top_pose = dataset_pb2.MatrixFloat()
                range_image_top_pose.ParseFromString(
                    zlib.decompress(laser.ri_return1.range_image_pose_compressed))
                range_image_pose = np.array(range_image_top_pose.data).reshape(range_image_top_pose.shape.dims)
                
            camera_projection = dataset_pb2.MatrixInt32()
            camera_projection.ParseFromString(
                    zlib.decompress(laser.ri_return1.camera_projection_compressed))
            camera_projection = np.array(camera_projection.data).reshape(camera_projection.shape.dims)

    else:
        # Return the second strongest response if available

        if len(laser.ri_return2.range_image_compressed) > 0:
            ri = dataset_pb2.MatrixFloat()
            ri.ParseFromString(
                zlib.decompress(laser.ri_return2.range_image_compressed))
            ri = np.array(ri.data).reshape(ri.shape.dims)
                
            camera_projection = dataset_pb2.MatrixInt32()
            camera_projection.ParseFromString(
                    zlib.decompress(laser.ri_return2.camera_projection_compressed))
            camera_projection = np.array(camera_projection.data).reshape(camera_projection.shape.dims)

    return ri, camera_projection, range_image_pose

def compute_beam_inclinations(calibration, height):
    """ Compute the inclination angle for each beam in a range image. """

    if len(calibration.beam_inclinations) > 0:
        return np.array(calibration.beam_inclinations)
    else:
        inclination_min = calibration.beam_inclination_min
        inclination_max = calibration.beam_inclination_max

        return np.linspace(inclination_min, inclination_max, height)

def compute_range_image_polar(range_image, extrinsic, inclination):
    """ Convert a range image to polar coordinates. """

    height = range_image.shape[0]
    width = range_image.shape[1]

    az_correction = math.atan2(extrinsic[1,0], extrinsic[0,0])
    azimuth = np.linspace(np.pi,-np.pi,width) - az_correction

    azimuth_tiled = np.broadcast_to(azimuth[np.newaxis,:], (height,width))
    inclination_tiled = np.broadcast_to(inclination[:,np.newaxis],(height,width))

    return np.stack((azimuth_tiled,inclination_tiled,range_image))

def compute_range_image_cartesian(range_image_polar, extrinsic, pixel_pose, frame_pose):
    """ Convert polar coordinates to cartesian coordinates. """

    azimuth = range_image_polar[0]
    inclination = range_image_polar[1]
    range_image_range = range_image_polar[2]

    cos_azimuth = np.cos(azimuth)
    sin_azimuth = np.sin(azimuth)
    cos_incl = np.cos(inclination)
    sin_incl = np.sin(inclination)

    x = cos_azimuth * cos_incl * range_image_range
    y = sin_azimuth * cos_incl * range_image_range
    z = sin_incl * range_image_range

    range_image_points = np.stack([x,y,z,np.ones_like(z)])

    range_image_points = np.einsum('ij,jkl->ikl', extrinsic,range_image_points)

    # TODO: Use the pixel_pose matrix. It seems that the bottom part of the pixel pose
    #       matrix is missing. Not sure if this is a bug in the dataset.

    #if pixel_pose is not None:
    #    range_image_points = np.einsum('hwij,jhw->ihw', pixel_pose, range_image_points)
    #    frame_pos_inv = np.linalg.inv(frame_pose)
    #    range_image_points = np.einsum('ij,jhw->ihw',frame_pos_inv,range_image_points)

        
    return range_image_points


def project_to_pointcloud(frame, ri, camera_projection, range_image_pose, calibration):
    """ Create a pointcloud in vehicle space from LIDAR range image. """
    beam_inclinations = compute_beam_inclinations(calibration, ri.shape[0])
    beam_inclinations = np.flip(beam_inclinations)

    extrinsic = np.array(calibration.extrinsic.transform).reshape(4,4)
    frame_pose = np.array(frame.pose.transform).reshape(4,4)

    ri_polar = compute_range_image_polar(ri[:,:,0], extrinsic, beam_inclinations)

    if range_image_pose is None:
        pixel_pose = None
    else:
        pixel_pose = get_rotation_matrix(range_image_pose[:,:,0], range_image_pose[:,:,1], range_image_pose[:,:,2])
        translation = range_image_pose[:,:,3:]
        pixel_pose = np.block([
            [pixel_pose, translation[:,:,:,np.newaxis]],
            [np.zeros_like(translation)[:,:,np.newaxis],np.ones_like(translation[:,:,0])[:,:,np.newaxis,np.newaxis]]])


    ri_cartesian = compute_range_image_cartesian(ri_polar, extrinsic, pixel_pose, frame_pose)
    ri_cartesian = ri_cartesian.transpose(1,2,0)

    mask = ri[:,:,0] > 0

    return ri_cartesian[mask,:3], ri[mask]


def get(object_list, name):
    """ Search for an object by name in an object list. """

    object_list = [obj for obj in object_list if obj.name == name]
    return object_list[0]

