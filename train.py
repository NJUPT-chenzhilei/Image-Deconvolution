""" <train.py>  Copyright (C) <2020>  <Yu Shi>
    This program comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
    This is free software, and you are welcome to redistribute it
    under certain conditions; type `show c' for details."""

import os
import torch.nn

"""global training parameters"""
use_CUDA = False
batch_size = 20
init_learning_rate = 0.001
exponent_decay_factor = 0.2
model_saving_path = './saved_model'


"""judge if GPU(s) is(are) available"""
if use_CUDA:
    use_CUDA = torch.cuda.is_available()
else:
    print('Your CUDA device(s) is(are) not available! We will use CPU to compute.')


def train():
    pass


if __name__ == '__main__':
    train()
