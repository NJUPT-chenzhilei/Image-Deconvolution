""" <train.py>  Copyright (C) <2020>  <Yu Shi>
    This program comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
    This is free software, and you are welcome to redistribute it
    under certain conditions; type `show c' for details."""


import os
import torch.nn as nn
import torch.cuda
import visdom
import numpy as np
import torch.optim as optim
import backbone
from data_provider import MyDataset
from skimage.measure import compare_psnr, compare_ssim
from torch.utils.data import DataLoader


"""global training parameters"""
use_CUDA = False
batch_size = 1
init_learning_rate = 0.001
exponent_decay_factor = 0.2
model_saving_path = './saved_model'
train_data_path = './data/train'
test_data_path = './data/test'
max_epoch_num = 200
iteration_num = 40
base_loss_function = nn.MSELoss()
print_per_n_batches = 1

"""judge if GPU(s) is(are) available"""
if use_CUDA:
    use_CUDA = torch.cuda.is_available()
else:
    print('Your CUDA device(s) is(are) not available! We will use CPU to train.')


def lr_scheduler(optimizer, epoch):
    """Decay learning rate by a factor of 0.5 every 5000."""
    if epoch % 50 == 0 and epoch > 55:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
        print('LR is set to {}'.format(param_group['lr']))


def compute_loss(input, target):
    return base_loss_function(input, target)


def train(model, epoch, optimizer, data_loader):
    """training function"""

    """enter train mode"""
    model.train()

    for batch_idx, (data, target) in enumerate(data_loader):
        print('  iteration {} out of {} in training'.format(batch_idx, epoch))

        """migrate data and model to GPU if CUDA is available"""
        if use_CUDA:
            data, target = data.cuda(), target.cuda()
            model.cuda()

        """inference data in model"""
        reconstruction = model(data)

        """compute loss"""
        loss = compute_loss(reconstruction, target)

        """backward and optimize"""
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        """print training information"""
        if (batch_idx + 1) % print_per_n_batches:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\n'
                  .format(epoch, (batch_idx+1) * len(data), len(data_loader.dataset),
                          100. * (batch_idx+1) / len(data_loader), loss.data[0]))


def test(model, epoch, data_loader):
    """enter eval mode"""
    model.eval()

    test_loss = 0

    for data, target in data_loader:
        output = model(data)
        test_loss += compute_loss(output, target)

    print('After {} epoch, Average loss on test set: {:.4f}.'
          .format(epoch, test_loss / len(data_loader)))


if __name__ == '__main__':
    """judge if directories are available"""
    if not os.path.exists(model_saving_path):
        os.mkdir(model_saving_path)

    """create backbone"""
    model = backbone.DCSC()

    """create dataloader"""
    train_data_loader = DataLoader(MyDataset(train_data_path))
    test_data_loader = DataLoader(MyDataset(test_data_path))

    """link to visdom"""
    # vis = visdom.Visdom()

    """optimizer"""
    optimizer = optim.Adam(model.parameters(), init_learning_rate)

    """train for every epoch"""
    for epoch in range(max_epoch_num):
        print('current_epoch:%d' % (epoch + 1))
        train(model, epoch, optimizer, train_data_loader)
        test(model, epoch, test_data_loader)
        lr_scheduler(optimizer, epoch)
