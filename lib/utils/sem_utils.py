import torch
import cv2
import numpy as np
from imgviz import label_colormap
from lib.config import cfg
from bidict import bidict


num_classes = cfg.data.get('num_classes', 0) + 1
default_colormap = label_colormap(num_classes)
def get_labe2color():
    pass

# From predicted semantic label to semantic map
# Either raw semantic logits or semantic probabilities(softmax)
def vis_semantic_label(semantics):
    '''
    semantic: [S, H, W]
    output: [H, W, 3]
    '''
    
    semantic_label = np.argmax(semantics, axis=0) # [H, W]
    colormap = get_labe2color()
    semantic_map = colormap[semantic_label].astype(np.uint8)
    semantic_map = semantic_map[..., [2, 1, 0]] # BGR to RGB
    return semantic_map

# From ground truth semantic label to semantic map
def vis_semantic_gt(semantic_gt):
    '''
    semantic_gt: [1, H, W]
    output: [H, W, 3]
    '''
    semantic_gt = semantic_gt.squeeze(0)
    valid_mask = (semantic_gt >= 0)
    _, h, w = semantic_gt.shape
    semantic_map = np.zeros((h, w, 3)).astype(np.uint8)
    colormap = get_labe2color()
    semantic_map[valid_mask] = colormap[semantic_gt[valid_mask]]
    return semantic_map

# From semantic map to semantic label
def get_semantic_label(semantic_path):
    semantic = cv2.imread(semantic_path)
    label2color = get_labe2color()
    h, w = semantic.shape[:2]
    semantic_flat = semantic.reshape(-1, 3).astype(np.uint8)
    semantic_label = np.zeros(semantic_flat.shape[0]).astype(np.float32)
    for label, color in label2color.items():
        color = np.array(color).astype(np.uint8)
        mask = np.all(semantic_flat == color, axis=-1)
        semantic_label[mask] = label
    semantic_label = semantic_label.reshape(h, w, -1)
    return semantic_label

