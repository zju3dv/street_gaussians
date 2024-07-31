import torch
import imageio
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
from PIL import Image

def save_img_torch(x, name='out.png'):
    x = (x.clamp(0., 1.).detach().cpu().numpy() * 255).astype(np.uint8)
    if x.shape[0] == 1 or x.shape[0] == 3:
        x = x.transpose(1, 2, 0)
    if x.shape[-1] == 1:
        x = x.squeeze(-1)
    
    img = Image.fromarray(x)
    img.save(name)
    
def save_img_numpy(x, name='out.png'):
    x = (x.clip(0., 1.) * 255).astype(np.uint8)
    if x.shape[0] == 1 or x.shape[0] == 3:
        x = x.transpose(1, 2, 0)
    if x.shape[-1] == 1:
        x = x.squeeze(-1)
    
    img = Image.fromarray(x)
    img.save(name)

def unnormalize_img(img, mean, std):
    """
    img: [3, h, w]
    """
    img = img.detach().cpu().clone()
    # img = img / 255.
    img *= torch.tensor(std).view(3, 1, 1)
    img += torch.tensor(mean).view(3, 1, 1)
    min_v = torch.min(img)
    img = (img - min_v) / (torch.max(img) - min_v)
    return img


def bgr_to_rgb(img):
    if img.shape[-1] == 3:
        return img[..., [2, 1, 0]]
    elif img.shape[-1] == 1:
        return np.repeat(img, 3, axis=-1)
    else:
        raise NotImplementedError

def rgb_to_bgr(img):
    if img.shape[-1] == 3:
        return img[..., [2, 1, 0]]
    elif img.shape[-1] == 1:
        return np.repeat(img, 3, axis=-1)
    else:
        raise NotImplementedError

to8b = lambda x: (255*np.clip(x, 0, 1)).astype(np.uint8)

def horizon_concate(inp0, inp1):
    h0, w0 = inp0.shape[:2]
    h1, w1 = inp1.shape[:2]
    if inp0.ndim == 3:
        inp = np.zeros((max(h0, h1), w0 + w1, 3), dtype=inp0.dtype)
        inp[:h0, :w0, :] = inp0
        inp[:h1, w0:(w0 + w1), :] = inp1
    else:
        inp = np.zeros((max(h0, h1), w0 + w1), dtype=inp0.dtype)
        inp[:h0, :w0] = inp0
        inp[:h1, w0:(w0 + w1)] = inp1
    return inp


def recover_shape(pixel_value, H, W, mask_at_box=None):
    from lib.config import cfg
    if pixel_value.shape[0] == 1:
        pixel_value = pixel_value[0]
    
    if len(pixel_value.shape) == 3:
        if pixel_value.shape[0] in [1, 3]:
            pixel_value = pixel_value.permute(1, 2, 0).detach().cpu().numpy()
        else:
            pixel_value = pixel_value.detach().cpu().numpy()

    elif len(pixel_value.shape) == 2:
        pixel_value = pixel_value.detach().cpu().numpy()
        pixel_value = pixel_value.reshape(H, W, -1)
        
    elif len(pixel_value.shape) == 1:
        pixel_value = pixel_value.detach().cpu().numpy()[..., None]
        pixel_value = pixel_value.reshape(H, W, -1)
    
    else:
        raise ValueError('Invalid shape of pixel_value: {}'.format(pixel_value.shape))

    
    if mask_at_box is not None:
        mask_at_box = mask_at_box.reshape(H, W)     
        
        full_pixel_value = np.ones((H, W, 3)).astype(np.float32) if cfg.white_bkgd else np.zeros((H, W, 3)).astype(np.float32)
        
        full_pixel_value[mask_at_box] = pixel_value
        return full_pixel_value
    else:
        # pixel_value = np.repeat(pixel_value, 3, axis=-1)
        return pixel_value
    

    
    
def vertical_concate(inp0, inp1):
    h0, w0 = inp0.shape[:2]
    h1, w1 = inp1.shape[:2]
    if inp0.ndim == 3:
        inp = np.zeros((h0 + h1, max(w0, w1), 3), dtype=inp0.dtype)
        inp[:h0, :w0, :] = inp0
        inp[h0:(h0 + h1), :w1, :] = inp1
    else:
        inp = np.zeros((h0 + h1, max(w0, w1)), dtype=inp0.dtype)
        inp[:h0, :w0] = inp0
        inp[h0:(h0 + h1), :w1] = inp1
    return inp


def save_image(pred, gt, save_dir, save_name, concat=False):
    if gt is None:
        cv2.imwrite('{}/{}.png'.format(save_dir, save_name), to8b(rgb_to_bgr(pred)))
    else:
        if concat:
            img = horizon_concate(pred, gt)
            cv2.imwrite('{}/{}.png'.format(save_dir, save_name), to8b(rgb_to_bgr(img)))
        else: 
            cv2.imwrite('{}/{}.png'.format(save_dir, save_name), to8b(rgb_to_bgr(pred)))
            cv2.imwrite('{}/{}_gt.png'.format(save_dir, save_name), to8b(rgb_to_bgr(gt)))
    
def transparent_cmap(cmap):
    """Copy colormap and set alpha values"""
    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = 0.3
    return mycmap

cmap = transparent_cmap(plt.get_cmap('jet'))


def set_grid(ax, h, w, interval=8):
    ax.set_xticks(np.arange(0, w, interval))
    ax.set_yticks(np.arange(0, h, interval))
    ax.grid()
    ax.set_yticklabels([])
    ax.set_xticklabels([])


color_list = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.167, 0.000, 0.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        1.000, 1.000, 1.000,
        0.50, 0.5, 0
    ]
).astype(np.float32)
colors = color_list.reshape((-1, 3)) * 255
colors = np.array(colors, dtype=np.uint8).reshape(len(colors), 1, 1, 3)

def visualize_depth_numpy(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """    
    x = np.nan_to_num(depth) # change nan to 0
    if minmax is None:
        mi = np.min(x[x>0]) # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi,ma = minmax
    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)
    return x_, [mi,ma]

def normalize_img(img):
    min, max = img.min(), img.max()
    img = (img - min) / (max - min)
    img = (img * 255).astype(np.uint8)
    return img

def linear_to_srgb(linear, eps=None):
    """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
    if eps is None:
        eps = torch.finfo(linear.dtype).eps
        # eps = 1e-3

    srgb0 = 323 / 25 * linear
    srgb1 = (211 * linear.clamp_min(eps) ** (5 / 12) - 11) / 200
    return torch.where(linear <= 0.0031308, srgb0, srgb1)

def srgb_to_linear(srgb, eps=None):
    """Assumes `srgb` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
    if eps is None:
        eps = np.finfo(srgb.dtype).eps
    linear0 = 25 / 323 * srgb
    linear1 = np.maximum(eps, ((200 * srgb + 11) / (211))) ** (12 / 5)
    return np.where(srgb <= 0.04045, linear0, linear1)

def draw_3d_box_on_img(vertices, img, color=(255, 128, 128), thickness=1):
    # Draw the edges of the 3D bounding box
    for k in [0, 1]:
        for l in [0, 1]:
            for idx1,idx2 in [((0,k,l),(1,k,l)), ((k,0,l),(k,1,l)), ((k,l,0),(k,l,1))]:
                cv2.line(img, tuple(vertices[idx1]), tuple(vertices[idx2]), color, thickness)
    
    # Draw a cross on the front face to identify front & back.
    for idx1,idx2 in [((1,0,0),(1,1,1)), ((1,1,0),(1,0,1))]:
        cv2.line(img, tuple(vertices[idx1]), tuple(vertices[idx2]), color, thickness)