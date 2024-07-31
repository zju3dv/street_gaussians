import cv2
import os
import numpy as np
import argparse
import glob

parsers = argparse.ArgumentParser(description="KITTI-STEP to panoptic segmentation")
parsers.add_argument("--annotation_path", type=str, default="")
parsers.add_argument("--output_path", type=str, default="")
annotation_path, output_path = parsers.parse_args().annotation_path, parsers.parse_args().output_path

# Follow KITTI_STEP annotation
# Label Name	Label ID
# road	0
# sidewalk	1
# building	2
# wall	3
# fence	4
# pole	5
# traffic light	6
# traffic sign	7
# vegetation	8
# terrain	9
# sky	10
# person†	11
# rider	12
# car†	13
# truck	14
# bus	15
# train	16
# motorcycle	17
# bicycle	18
# void	255

colormap = np.zeros((256, 3), dtype=np.uint8)

colormap[0] = [128, 64, 128]
colormap[1] = [244, 35, 232]
colormap[2] = [70, 70, 70]
colormap[3] = [102, 102, 156]
colormap[4] = [190, 153, 153]
colormap[5] = [153, 153, 153]
colormap[6] = [70, 130, 180]
colormap[7] = [220, 220, 0]
colormap[8] = [107, 142, 35]
colormap[9] = [152, 251, 152]
colormap[10] = [250, 170, 30]
colormap[11] = [220, 20, 60]
colormap[12] = [255, 0, 0]
colormap[13] = [0, 0, 142]
colormap[14] = [0, 0, 70]
colormap[15] = [0, 60, 100]
colormap[16] = [0, 80, 100]
colormap[17] = [0, 0, 230]
colormap[18] = [119, 11, 32]
colormap[255] = [0, 0, 0]



mask_label_list = np.array([11, 12, 13, 14, 15, 16, 17, 18]).astype(np.uint8)

if __name__ == "__main__":
    args = parsers.parse_args()
    
    annotation_path = args.annotation_path
    output_paths = args.output_path
    
    os.makedirs(output_path, exist_ok=True)
    filenames = sorted(glob.glob(f'{annotation_path}/**/*.png', recursive=True))
    filenames = [os.path.relpath(filename, annotation_path) for filename in filenames]
    
    for filename in filenames:
        file_path = os.path.join(annotation_path, filename)
        img = cv2.imread(file_path)

        modified_img = img.copy()
        
        label = img[..., 2].astype(np.uint8)
        mask = np.isin(label, mask_label_list)
        
        modified_img = np.zeros_like(img).astype(np.uint8)
        modified_img[~mask] = 255        
        
        modified_file_path = os.path.join(output_path, filename)
        modifeid_file_folder = os.path.dirname(modified_file_path)
        if not os.path.exists(modifeid_file_folder):
            os.makedirs(modifeid_file_folder, exist_ok=True)
        
        print(modified_file_path)
        cv2.imwrite(modified_file_path, modified_img)