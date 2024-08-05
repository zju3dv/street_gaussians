import numpy as np
import torch
import os
import math
from lib.utils.graphics_utils import focal2fov
from lib.datasets.base_readers import CameraInfo
from PIL import Image
from tqdm import tqdm

def to_cuda(batch):
    if isinstance(batch, tuple) or isinstance(batch, list):
        batch = [to_cuda(b) for b in batch]
        return batch
    elif isinstance(batch, torch.Tensor):
        return batch.cuda()
    elif isinstance(batch, np.ndarray):
        return torch.from_numpy(batch).cuda()
    elif isinstance(batch, dict):
        for k in batch:
            if k == 'meta':
                continue
            batch[k] = to_cuda(batch[k])
        return batch
    else:
        raise NotImplementedError

def get_split_data(split_train, split_test, data):
    if split_train != -1:
        train_data = [d for idx, d in enumerate(data) if idx % split_train == 0]
        test_data = [d for idx, d in enumerate(data) if idx % split_train != 0]
    else:
        train_data = [d for idx, d in enumerate(data) if idx % split_test != 0]
        test_data = [d for idx, d in enumerate(data) if idx % split_test == 0]
    return train_data, test_data

def get_val_frames(num_frames: int, test_every: int, train_every: int):
    if train_every is None or train_every < 0:
        val_frames = set(np.arange(test_every, num_frames, test_every))
        train_frames = (set(np.arange(num_frames)) - val_frames) if test_every > 1 else set()
    else:
        train_frames = set(np.arange(0, num_frames, train_every))
        val_frames = (set(np.arange(num_frames)) - train_frames) if train_every > 1 else set()

    train_frames = sorted(list(train_frames))
    val_frames = sorted(list(val_frames))

    return train_frames, val_frames



