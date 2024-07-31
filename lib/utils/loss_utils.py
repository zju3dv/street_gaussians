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

import torch
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from math import exp
from lib.utils.img_utils import save_img_torch
from lib.config import cfg


def l1_loss(network_output, gt, mask=None):
    '''
    network_output, gt: (C, H, W)
    mask: (1, H, W) 
    '''

    network_output = network_output.permute(1, 2, 0) # [H, W, C]
    gt = gt.permute(1, 2, 0) # [H, W, C]

    if mask is not None:
        mask = mask.squeeze(0) # [H, W]
        network_output = network_output[mask]
        gt = gt[mask]
    
    loss = ((torch.abs(network_output - gt))).mean()

    return loss

def l2_loss(network_output, gt, mask=None):
    '''
    network_output, gt: (C, H, W)
    mask: (1, H, W) 
    '''
    
    network_output = network_output.permute(1, 2, 0) # [H, W, C]
    gt = gt.permute(1, 2, 0) # [H, W, C]    
    
    if mask is not None:
        mask = mask.squeeze(0) # [H, W]
        network_output = network_output[mask]
        gt = gt[mask]

    loss =  (((network_output - gt) ** 2)).mean()

    return loss

    
def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2, mask=None):
    '''
    img1, img2: (C, H, W)
    mask: (1, H, W)
    '''    
    
    img1 = img1.permute(1, 2, 0)
    img2 = img2.permute(1, 2, 0)
    
    if mask is not None:
        mask = mask.squeeze(0)
        img1 = img1[mask]
        img2 = img2[mask]
    
    # mse = ((img1 - img2) ** 2).view(-1, img1.shape[-1]).mean(dim=0, keepdim=True)    
    mse = torch.mean((img1 - img2) ** 2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr
    
    
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True, mask=None):
    channel = img1.size(-3)
    window = create_window(window_size, channel)
    
    if mask is not None:
        img1 = torch.where(mask, img1, torch.zeros_like(img1))
        img2 = torch.where(mask, img2, torch.zeros_like(img2))
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())

    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)