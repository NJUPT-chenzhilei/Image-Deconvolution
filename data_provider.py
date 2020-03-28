""" <data_provider.py>  Copyright (C) <2020>  <Yu Shi>
    This program comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
    This is free software, and you are welcome to redistribute it
    under certain conditions; type `show c' for details."""


import os
import torch
import skimage.io as io
import numpy as np
from torch.utils.data import DataLoader, Dataset
import cv2

"""global parameters"""
compressed_level = 3


def toTensor(img):
    assert type(img) == np.ndarray, 'the img type is {}, but ndarry expected'.format(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    img = img.float().div(255)
    return img


class MyDataset(Dataset):
    """Inherit torch.utils.data.dataset"""
    def __init__(self, file_path):
        super().__init__()

        self.file_path = file_path
        self.length = len(os.listdir(os.path.join(self.file_path, 'raw')))
        """read dataset from file"""

    def __len__(self):
        return compressed_level * self.length

    def __getitem__(self, item):
        """Get data with index of item."""

        dir_index = item // self.length
        data_index = item % self.length

        dir_name = os.path.join(self.file_path, 'compressed_' + str(dir_index + 1))
        data_name = os.path.join(dir_name, os.listdir(dir_name)[data_index])
        target_name = os.path.join(os.path.join(self.file_path, 'raw'), os.listdir(dir_name)[data_index])

        data_array = cv2.imread(data_name)
        target_array = cv2.imread(target_name)
        return toTensor(data_array), toTensor(target_array)
