""" <evaluate.py>  Copyright (C) <2020>  <Yu Shi>
    This program comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
    This is free software, and you are welcome to redistribute it
    under certain conditions; type `show c' for details."""

import os
import torch.nn as nn
import torch.cuda

"""global evaluating parameters"""
use_CUDA = False
batch_size = 20
model_path = './saved_model'


"""judge if GPU(s) is(are) available"""
if use_CUDA:
    use_CUDA = torch.cuda.is_available()
else:
    print('Your CUDA device(s) is(are) not available! We will use CPU to compute.')


def evaluate():
    """evaluate mode"""


    """load saved model"""
    if not os.path.exists(model_path):
        print('No model named %s' % model_path)
        exit(-1)



if __name__ == '__main__':
    evaluate()
