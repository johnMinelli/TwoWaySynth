from __future__ import print_function
import numpy as np
import os
from numpy.lib.stride_tricks import as_strided
from PIL import Image, ImageDraw
import random
import torch
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

def fix_random(seed: int) -> None:
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


def tensor2im(tensor, colormap='rainbow', imtype=np.uint8):
    assert (tensor.ndimension() <= 3), "Tensor should have 3 or less dimensions, remove the batch dimension"
    tensor = tensor.detach().cpu().float()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        array = tensor.squeeze().numpy()
        norm_array = (array - array.min()) / (array.max() - array.min() + 1.e-17) if np.any(array) else array
        array = COLORMAPS[colormap](norm_array).astype(np.float32) * 255.
    elif tensor.ndimension() == 3:
        assert (tensor.size(0) == 3)
        array = tensor.numpy() * 0.5 + 0.5
        array = array.transpose((1, 2, 0)) * 255.
    array = array.astype(imtype).copy()
    return array


# def tensor2im(image_tensor, imtype=np.uint8):
#     image_numpy = image_tensor[0].detach().cpu().float().numpy()
#     if image_numpy.ndim == 2:
#         image_numpy = image_numpy.reshape((1,image_numpy.shape[0],image_numpy.shape[1]))
#     if image_numpy.shape[0] == 1:
#         image_numpy = np.tile(image_numpy, (3, 1, 1))
# 
#     image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
#     image_numpy = image_numpy.astype(imtype).copy()
# 
#     return image_numpy

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

def tile_array(a, b0, b1):
    r, c = a.shape                                    # number of rows/columns
    rs, cs = a.strides                                # row/column strides
    x = as_strided(a, (r, b0, c, b1), (rs, 0, cs, 0)) # view a as larger 4D array
    return x.reshape(r*b0, c*b1)


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
             'magma': high_res_colormap(cm.get_cmap('magma')),
             'bone': cm.get_cmap('bone', 10000)}

