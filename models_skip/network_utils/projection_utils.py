import numpy as np
from util_skip.util import tensor2im
from skimage.transform import warp
import torch


def get_RT(vec, dataset_type):
    from scipy.spatial.transform import Rotation as ROT
    if dataset_type in ['shapenet']:
        T = np.array([0, 0, 2]).reshape((3, 1))
        R = ROT.from_euler('xyz', vec[:3]).as_matrix()
        T = -R.dot(T) + T
    else:
        R = ROT.from_euler('xyz', vec[0:3]).as_matrix()
        T = vec[3:].reshape((3, 1))
    mat = np.block([[R, T], [np.zeros((1, 3)), 1]])

    return torch.Tensor(mat).float().unsqueeze(0)

def scale_K(intrinsics, scale):
    #scale fx, fy, cx, cy according to the scale

    K = intrinsics.clone()
    K[:, 0, 0] *= scale
    K[:, 1, 1] *= scale
    K[:, 0, 2] *= scale
    K[:, 1, 2] *= scale
    return K

def scale_image(batch_img, scale_x, scale_y):
    batch_size = batch_img.size()[0]
    scale_mat = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).numpy()
    scale_mat[:, 0, 0] = scale_x
    scale_mat[:, 0, 2] = (batch_img.size()[3] - (batch_img.size()[3] * scale_x)) / 2
    scale_mat[:, 1, 1] = scale_y
    scale_mat[:, 1, 2] = (batch_img.size()[2] - (batch_img.size()[2] * scale_y)) / 2
    scaled_image = torch.from_numpy(np.array([warp(tensor2im(img), np.linalg.inv(sc)) for img, sc in
                                            zip(batch_img, scale_mat)]).transpose((0, 3, 1, 2)))
    return scaled_image


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rotation_matrix_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rotation_matrix_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot
