import imageio
import os
import numpy as np
import argparse
import glob

parsers = argparse.ArgumentParser(description="KITTI-STEP to panoptic segmentation")
parsers.add_argument("--annotation_path", type=str, default="")
parsers.add_argument("--output_path", type=str, default="")
annotation_path, output_path = parsers.parse_args().annotation_path, parsers.parse_args().output_path


colormap = np.zeros((256, 3), dtype=np.uint8)

colormap[0] = [128, 64, 128]
colormap[1] = [232, 35, 244]
colormap[2] = [70, 70, 70]
colormap[3] = [156, 102, 102]
colormap[4] = [153, 153, 190]
colormap[5] = [153, 153, 153]
colormap[6] = [180, 130, 70]
colormap[7] = [0, 220, 220]
colormap[8] = [35, 142, 107]
colormap[9] = [152, 251, 152]
colormap[10] = [30, 170, 250]
colormap[11] = [60, 20, 220]
colormap[12] = [0, 0, 255]
colormap[13] = [142, 0, 0]
colormap[14] = [70, 0, 0]
colormap[15] = [100, 60, 0]
colormap[16] = [100, 80, 0]
colormap[17] = [230, 0, 0]
colormap[18] = [32, 11, 119]
colormap[255] = [0, 0, 0]

    
if __name__ == "__main__":
    args = parsers.parse_args()
    
    annotation_path = args.annotation_path
    output_paths = args.output_path
    
    os.makedirs(output_path, exist_ok=True)
    filenames = sorted(glob.glob(f'{annotation_path}/**/*.png', recursive=True))
    filenames = [os.path.relpath(filename, annotation_path) for filename in filenames]
    
    for filename in filenames:
        file_path = os.path.join(annotation_path, filename)
        img = imageio.imread(file_path)
        modified_img = img.copy()
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                pixel = img[i, j]
                r, g, b = pixel[0], pixel[1], pixel[2]
                modified_img[i, j] = colormap[r]

        modified_file_path = os.path.join(output_path, filename)
        modifeid_file_folder = os.path.dirname(modified_file_path)
        if not os.path.exists(modifeid_file_folder):
            os.makedirs(modifeid_file_folder, exist_ok=True)
        
        print(modified_file_path)
        imageio.imwrite(modified_file_path, modified_img)