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
use_CUDA = True
batch_size = 16
init_learning_rate = 0.001
exponent_decay_factor = 0.2
model_saving_path = './saved_model'
train_data_path = './data/train'
test_data_path = './data/test'
max_epoch_num = 200
iteration_num = 40
base_loss_function = nn.MSELoss()
print_per_n_batches = 2

"""judge if GPU(s) is(are) available"""
if use_CUDA:
    use_CUDA = torch.cuda.is_available()
else:
    print('Your CUDA device(s) is(are) not available! We will use CPU to train.')
device = torch.device("cuda:1" if use_CUDA else 'cpu')


def lr_scheduler(optimizer, epoch):
    """Decay learning rate by a factor of 0.5 every 5000."""
    if epoch % 50 == 0 and epoch > 55:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
        print('LR is set to {}'.format(param_group['lr']))


def compute_loss(input, target):
    total_loss = 0
    loss_function = [base_loss_function, base_loss_function, base_loss_function, base_loss_function, base_loss_function]
    for i in range(len(input)):
        total_loss += loss_function[i](input[i], target[i])
    return total_loss


def train(model, epoch, optimizer, data_loader, kernel):
    """training function"""

    """enter train mode"""
    model.train()

    for batch_idx, (data, target) in enumerate(data_loader):
        print('iteration {} out of {} in training'.format(batch_idx + 1, epoch + 1))

        """migrate data and model to GPU if CUDA is available"""
        if use_CUDA:
            data = data.to(device)
            target = target.to(device)
            model = model.to(device)

        """inference data in model"""
        reconstruction = model(data)

        """compute Gaussian pyramids of target"""
        gp = backbone.GaussianPyramid(kernel)
        target = gp(target)

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
                          100. * (batch_idx+1) / len(data_loader), loss.data))


def test(model, epoch, data_loader, kernel):
    """enter eval mode"""
    model.eval()

    test_loss = 0

    for data, target in data_loader:
        """migrate data and model to GPU if CUDA is available"""
        if use_CUDA:
            data = data.to(device)
            target = target.to(device)
            model = model.to(device)

        """compute Gaussian pyramids of target"""
        gp = backbone.GaussianPyramid(kernel)
        target = gp(target)
        output = model(data)

        test_loss += compute_loss(output, target)

    print('After {} epoch, Average loss on test set: {:.4f}.'
          .format(epoch, test_loss / len(data_loader)))


if __name__ == '__main__':
    """judge if directories are available"""
    if not os.path.exists(model_saving_path):
        os.mkdir(model_saving_path)

    """generate Gaussian kernel"""
    k = np.float32([.0625, .25, .375, .25, .0625])  # Gaussian kernel for image pyramid
    k = np.outer(k, k)
    # kernel = k[:, :, None, None] / k.sum() * np.eye(3, dtype=np.float32)
    kernel = k[:, :] / k.sum()
    kernel = np.expand_dims(kernel, 0).repeat(3, axis=0)
    kernel = np.expand_dims(kernel, 0).repeat(3, axis=0)
    # kernel = np.expand_dims(k, 0).repeat(3, axis=0)
    kernel = torch.from_numpy(kernel)
    kernel = kernel.to(device)

    """create backbone"""
    # model = backbone.DCSC()
    model = backbone.MyNetwork(kernel)

    """create dataloader"""
    train_data_loader = DataLoader(MyDataset(train_data_path), batch_size=batch_size, shuffle=True, num_workers=1)
    test_data_loader = DataLoader(MyDataset(test_data_path), batch_size=batch_size, shuffle=True, num_workers=1)

    """link to visdom"""
    # vis = visdom.Visdom()

    """optimizer"""
    optimizer = optim.Adam(model.parameters(), init_learning_rate)

    """train for every epoch"""
    for epoch in range(max_epoch_num):
        print('current_epoch:%d' % (epoch + 1))
        train(model, epoch, optimizer, train_data_loader, kernel)
        test(model, epoch, test_data_loader, kernel)
        lr_scheduler(optimizer, epoch)