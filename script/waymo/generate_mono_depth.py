import torch
import argparse
import os
import sys
import imageio
import numpy as np
sys.path.append(os.getcwd())
from glob import glob
from tqdm import tqdm
from lib.utils.img_utils import visualize_depth_numpy

# TODO: try other depth estimation models (Marigold, DepthAnything etc.)

midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
for param in midas.parameters():
    param.requires_grad = False

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform
downsampling = 1

def estimate_depth(img, mode='test'):
    h, w = img.shape[1:3]
    norm_img = (img[None] - 0.5) / 0.5
    norm_img = torch.nn.functional.interpolate(
        norm_img,
        size=(384, 512),
        mode="bicubic",
        align_corners=False)

    if mode == 'test':
        with torch.no_grad():
            prediction = midas(norm_img)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(h//downsampling, w//downsampling),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
    else:
        prediction = midas(norm_img)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(h//downsampling, w//downsampling),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    return prediction

def run_midas(images_lists, output_dir, ignore_exists):
    save_dir = os.path.join(output_dir, f'midas_depth')
    images_lists = sorted(images_lists) 

    os.makedirs(save_dir, exist_ok=True)
    for image_path in tqdm(images_lists):
        image_base_name = os.path.basename(image_path)
        output_depth_vis = os.path.join(save_dir, image_base_name)
        output_depth = os.path.join(save_dir, image_base_name.replace('.png', '.npy'))
        
        if os.path.exists(output_depth) and ignore_exists:
            print(f'{output_depth} exists, skip')
            continue
        
        image = imageio.imread(image_path) / 255.
        image = torch.from_numpy(image).permute(2, 0, 1).float().cuda()
        midas_depth = estimate_depth(image)
        midas_depth = midas_depth.cpu().numpy()
        midas_depth_vis = visualize_depth_numpy(midas_depth)[0][..., [2, 1, 0]]
        np.save(output_depth, midas_depth)
        imageio.imwrite(output_depth_vis, midas_depth_vis)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument("--ignore_exists", action='store_true')

    args = parser.parse_args()

    image_files = glob(args.input_dir + "/*.png")    
    
    run_midas(
        images_lists=image_files,
        output_dir=args.output_dir,
        ignore_exists=args.ignore_exists
    )