#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import torchvision.transforms.functional as tf
import json
from PIL import Image
from pathlib import Path

from tqdm import tqdm
from lib.config import cfg
from lib.utils.loss_utils import ssim, psnr
from lib.utils.lpipsPyTorch import lpips
from lib.datasets.dataset import Dataset


def evaluate(split='test'):
    scene_dir = cfg.model_path
    dataset = Dataset()
    if split == 'test':
        test_dir = Path(scene_dir) / "test"
        cam_infos = dataset.test_cameras[1]
    else:
        test_dir = Path(scene_dir) / "train"
        cam_infos = dataset.train_cameras[1]
        
    cam_infos = list(sorted(cam_infos, key=lambda x: x.id))
    
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    
    print(f"Scene: {scene_dir }")
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}
    
    for method in os.listdir(test_dir):
        print("Method:", method)
        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}    


        renders = []
        gts = []
        image_names = []

        for cam_info in tqdm(cam_infos, desc="Reading image progress"):
            image_name = cam_info.image_name
            render_path = test_dir / method / f'{image_name}_rgb.png'
            gt_path = test_dir / method / f'{image_name}_gt.png'
            
            render = Image.open(render_path)
            gt = Image.open(gt_path)
            renders.append(tf.to_tensor(render)[:3, :, :])
            gts.append(tf.to_tensor(gt)[:3, :, :])
            image_names.append(image_name)

        psnrs = []
        ssims = []
        lpipss = []

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            render = renders[idx].cuda()
            gt = gts[idx].cuda()
            ssims.append(ssim(render, gt))
            psnrs.append(psnr(render, gt))
            lpipss.append(lpips(render, gt, net_type='alex'))
        
        print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
        print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
        print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
        print("")
        
        full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                        "LPIPS": torch.tensor(lpipss).mean().item()})
        per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

    with open(scene_dir + f"/results_{split}.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + f"/per_view_{split}.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)

if __name__ == "__main__":
    if cfg.eval.eval_train:
        evaluate(split='train')
    if cfg.eval.eval_test:
        evaluate(split='test')
