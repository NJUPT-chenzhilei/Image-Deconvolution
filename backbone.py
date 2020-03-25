""" <backbone.py>  Copyright (C) <2020>  <Yu Shi>
    This program comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
    This is free software, and you are welcome to redistribute it
    under certain conditions; type `show c' for details."""


import torch.nn as nn
from collections import OrderedDict

"""global neural network parameters"""
image_channels = 3
conv_channels = 128
iterations = 40


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
            output_tensor += input_tensor
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


"""Source TensorFlow Code"""
# def DeepCSC(J, num_features=128, iterations=40):
#     _, _, _, channels = J.get_shape().as_list()
#
#     with tf.variable_scope("DCSC", reuse=tf.AUTO_REUSE):
#         with tf.variable_scope('X'):
#             X_1 = tf.layers.conv2d(J, 48, 3, dilation_rate=(1, 1), padding="SAME", name='X1')
#             X_2 = tf.layers.conv2d(J, 48, 3, dilation_rate=(2, 2), padding="SAME", name='X2')
#             X_4 = tf.layers.conv2d(J, 48, 3, dilation_rate=(4, 4), padding="SAME", name='X4')
#             X = tf.concat([X_1, X_2, X_4], -1)
#
#         with tf.variable_scope('G_X'):
#             G_X = tf.layers.conv2d(X, num_features, 3, padding='same', name='G_X', use_bias=False)
#
#         with tf.variable_scope('optimization'):
#             U = tf.nn.relu(G_X)
#             for _ in range(iterations):
#                 S_U = tf.layers.conv2d(U, num_features, 3, padding='same', name='S_U', use_bias=False)
#                 U = tf.nn.relu(G_X + S_U)
#
#         with tf.variable_scope('recons'):
#             H = tf.layers.conv2d(U, channels, 3, padding='same', name='residual')
#             O = J + H
#
#     return O


if __name__ == '__main__':
    network = DCSC()
    print(network)
