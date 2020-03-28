""" <backbone.py>  Copyright (C) <2020>  <Yu Shi>
    This program comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
    This is free software, and you are welcome to redistribute it
    under certain conditions; type `show c' for details."""


import torch.nn as nn
import numpy as np
from collections import OrderedDict

"""global neural network parameters"""
# number of channels of image
image_channels = 3
# DCSC network convolutional channels
conv_channels = 128
# number of DCSC network iterations
iterations = 40
# number of pyramid levels
num_pyramids = 5


class GaussianPyramid(nn.Module):
    """implementation of Gaussian Pyramid"""
    def __init__(self, kernel):
        super().__init__()

        self.kernel = kernel
        self.levels = num_pyramids

    def forward(self, input_tensor):
        feature_maps = [input_tensor]
        for i in range(self.levels - 1):
            input_tensor = nn.functional.conv2d(input_tensor, self.kernel, stride=(2, 2), bias=None, padding=1)
            feature_maps.append(input_tensor)
        return feature_maps[::-1]


class LaplacianPyramid(nn.Module):
    """implementation of Laplacian Pyramid"""
    def __init__(self, kernel, levels):
        super().__init__()

        self.kernel = kernel
        self.levels = levels

    def lap_split(self, img):
        """split and up-sample"""
        """shape amendment"""

        # print('kernel type is', end='')
        # print(self.kernel.is_cuda)
        # print('img type is', end='')
        # print(img.is_cuda)
        low = nn.functional.conv2d(img, self.kernel, stride=(2, 2), padding=1)
        # low = tf.nn.conv2d(img, kernel, [1, 2, 2, 1], 'SAME')
        # print(low.size()[0])
        if img.size()[3] % 2:
            low_upsample = nn.functional.conv_transpose2d(low, self.kernel * 4, stride=(2, 2), padding=1)
        else:
            low_upsample = nn.functional.conv_transpose2d(low, self.kernel * 4, stride=(2, 2), padding=1, output_padding=1)
        # low_upsample = shape_amendment(low)
        # low_upsample = tf.nn.conv2d_transpose(low, kernel * 4, tf.shape(img), [1, 2, 2, 1])
        high = img - low_upsample
        return low, high

    def forward(self, input_tensor):
        feature_maps = []
        img = input_tensor
        for i in range(self.levels):
            img, high = self.lap_split(img)
            feature_maps.append(high)
        feature_maps.append(img)
        return feature_maps[::-1]


class Optimization(nn.Module):
    """Implementation of Optimization part of DCSC"""
    def __init__(self):
        super().__init__()

        self.denoiser_1 = nn.Sequential(OrderedDict([
            ('S_U', nn.Conv2d(conv_channels, conv_channels, 3, padding=1, bias=False))]))
        self.denoiser_2 = nn.Sequential(OrderedDict([
            ('ReLU', nn.ReLU(inplace=True))]))

    def forward(self, input_tensor):
        for _ in range(iterations):
            output_tensor = self.denoiser_1(input_tensor)
            output_tensor = input_tensor + output_tensor
            output_tensor = self.denoiser_2(output_tensor)
        return output_tensor


class DCSC(nn.Module):
    """Implementation of Deep Convolutional Sparse Coding"""
    def __init__(self):
        super().__init__()

        self.G_X = nn.Conv2d(image_channels, conv_channels, 3, padding=1, bias=False)
        self.DCSC_network = nn.Sequential(OrderedDict([
            ('G_X', self.G_X),
            ('Optimization', Optimization()),
            ('Restore', nn.Conv2d(conv_channels, image_channels, 3, padding=1, bias=False))
        ]))

    def forward(self, input_tensor):
        return self.DCSC_network(input_tensor) + input_tensor


class MyNetwork(nn.Module):
    """framework of all network"""
    def __init__(self, kernel):
        super().__init__()

        self.kernel = kernel
        self.sub_net = DCSC()
        self.gaussian = GaussianPyramid(self.kernel)
        self.laplacian = LaplacianPyramid(self.kernel, num_pyramids - 1)

    def forward(self, input_tensor):
        output_pyramids = []
        upsample_sequence = []
        laplacian_pyramids = self.laplacian(input_tensor)
        for i in range(num_pyramids):
            output = laplacian_pyramids[i]
            output = self.sub_net(output) + output
            if not i == 0:
                output = upsample_sequence[i - 1] + output
            output = nn.functional.relu(output, inplace=True)
            output_pyramids.append(output)
            if not i == num_pyramids - 1:
                if laplacian_pyramids[i + 1].size()[3] % 2:
                    output = nn.functional.conv_transpose2d(output, self.kernel, stride=(2, 2), padding=1)
                else:
                    output = nn.functional.conv_transpose2d(output, self.kernel, stride=(2, 2), padding=1, output_padding=1)
                upsample_sequence.append(output)
        return output_pyramids


if __name__ == '__main__':
    network = DCSC()
    print(network)