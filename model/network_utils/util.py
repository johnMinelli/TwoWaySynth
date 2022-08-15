from __future__ import print_function
import numpy as np
import os
from numpy.lib.stride_tricks import as_strided
from PIL import Image, ImageDraw
import random
import torch
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from torch import nn
from torch.optim import lr_scheduler


def fix_random(seed: int):
    """Fix all the possible sources of randomness.
    Args:
        seed: the seed to use.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def tensor2im(tensor, colormap='viridis', imtype=np.uint8):
    assert (tensor.ndimension() <= 3), "Tensor should have 3 or less dimensions, remove the batch dimension"
    tensor = tensor.detach().cpu().float()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        array = tensor.squeeze().numpy()
        mask = array > 0
        if np.any(mask):
            min = array[mask].min()
            max = array[mask].max()
            array[np.invert(mask)] = min
            norm_array = (array - min) / (max - min + 1.e-17)
        else:
            norm_array = array
        array = COLORMAPS[colormap](norm_array).astype(np.float32) * 255.
        # array[np.invert(mask)] = 0
    elif tensor.ndimension() == 3:
        assert (tensor.size(0) == 3)
        array = tensor.numpy() * 0.5 + 0.5
        array = array.transpose((1, 2, 0)) * 255.
    array = array.astype(imtype).copy()
    return array


def tile_array(a, b0, b1):
    r, c = a.shape                                    # number of rows/columns
    rs, cs = a.strides                                # row/column strides
    x = as_strided(a, (r, b0, c, b1), (rs, 0, cs, 0)) # view a as larger 4D array
    return x.reshape(r*b0, c*b1)


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)



def get_scheduler(optimizer, opt, last_epoch):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 - opt.lr_niter_frozen) / float(opt.lr_niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=last_epoch)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_every, gamma=0.1, last_epoch=last_epoch)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5, last_epoch=last_epoch)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def high_res_colormap(low_res_cmap, resolution=1000, max_value=1):
    # Construct the list colormap, with interpolated values for higer resolution
    # For a linear segmented colormap, you can just specify the number of point in
    # cm.get_cmap(name, lutsize) with the parameter lutsize
    x = np.linspace(0,1,low_res_cmap.N)
    low_res = low_res_cmap(x)
    new_x = np.linspace(0,max_value,resolution)
    high_res = np.stack([np.interp(new_x, x, low_res[:,i]) for i in range(low_res.shape[1])], axis=1)
    return ListedColormap(high_res)


def opencv_rainbow(resolution=1000):
    # Construct the opencv equivalent of Rainbow
    opencv_rainbow_data = (
        (0.000, (1.00, 0.00, 0.00)),
        (0.400, (1.00, 1.00, 0.00)),
        (0.600, (0.00, 1.00, 0.00)),
        (0.800, (0.00, 0.00, 1.00)),
        (1.000, (0.60, 0.00, 1.00))
    )

    return LinearSegmentedColormap.from_list('opencv_rainbow', opencv_rainbow_data, resolution)


COLORMAPS = {'rainbow': opencv_rainbow(),
             'magma': high_res_colormap(cm.get_cmap('magma_r')),
             'plasma': high_res_colormap(cm.get_cmap('plasma_r')),
             'viridis': high_res_colormap(cm.get_cmap('viridis_r')),
             'bone': cm.get_cmap('bone', 10000)}

