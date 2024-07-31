import os
import random
import json
from lib.utils.camera_utils import camera_to_JSON, cameraList_from_camInfos
from lib.config import cfg
from lib.datasets.base_readers import storePly, SceneInfo
from lib.datasets.colmap_readers import readColmapSceneInfo
from lib.datasets.blender_readers import readNerfSyntheticInfo
from lib.datasets.waymo_full_readers import readWaymoFullInfo

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Waymo": readWaymoFullInfo,
}

class Dataset():
    def __init__(self):
        self.cfg = cfg.data
        self.model_path = cfg.model_path
        self.source_path = cfg.source_path
        self.images = self.cfg.images

        self.train_cameras = {}
        self.test_cameras = {}

        dataset_type = cfg.data.get('type', "Colmap")
        assert dataset_type in sceneLoadTypeCallbacks.keys(), 'Could not recognize scene type!'
        
        scene_info: SceneInfo = sceneLoadTypeCallbacks[dataset_type](self.source_path, **cfg.data)

        if cfg.mode == 'train':
            print(f'Saving input pointcloud to {os.path.join(self.model_path, "input.ply")}')
            pcd = scene_info.point_cloud
            storePly(os.path.join(self.model_path, "input.ply"), pcd.points, pcd.colors)

            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))

            print(f'Saving input camera to {os.path.join(self.model_path, "cameras.json")}')
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)
       
        self.scene_info = scene_info
        
        if self.cfg.shuffle and cfg.mode == 'train':
            random.shuffle(self.scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(self.scene_info.test_cameras)  # Multi-res consistent random shuffling
        
        for resolution_scale in cfg.resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(self.scene_info.train_cameras, resolution_scale)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(self.scene_info.test_cameras, resolution_scale)
            