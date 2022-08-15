import cv2
import numpy as np
import torch
import torch.nn.functional as F
from Forward_Warp import forward_warp

from model.network_utils.resample2d_package.resample2d import Resample2d

pixel_coords = None

def set_id_grid(depth):
    b, h, w = depth.size()
    i_range = torch.arange(0, h).view(1, h, 1).expand(1,h,w).type_as(depth)  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(1,h,w).type_as(depth)  # [1, H, W]
    ones = torch.ones(1,h,w).type_as(depth)

    return torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]

def check_sizes(input, input_name, expected):
    condition = [input.ndimension() == len(expected)]
    for i,size in enumerate(expected):
        if size.isdigit():
            condition.append(input.size(i) == int(size))
    assert(all(condition)), "wrong size for {}, expected {}, got  {}".format(input_name, 'x'.join(expected), list(input.size()))


def pixel2cam(input, depth, intrinsics_inv):

    """Transform input values in the pixel frame to the world camera frame.
    Args:
        depth: depth maps -- [B, H, W]
        intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
    Returns:
        array of cam points values -- [B, 3, H, W]
    """
    b, h, w = depth.size()
    current_pixel_input = input[:,:,:h,:w].expand(b,3,h,w).reshape(b, 3, -1)  # [B, 3, H*W]
    cam_input = (intrinsics_inv.bmm(current_pixel_input)).reshape(b, 3, h, w)

    d = depth.unsqueeze(1).clone()
    d[d<0] = 0
    # return torch.cat([cam * d, torch.ones(b,1,h,w).type_as(depth)], 1)  # add 1s to later make a bmm with full R mat
    return cam_input * d


def cam2pixel(cam, proj_c2p_rot, proj_c2p_tr, padding_mode='zeros', normalize=True):
    """Transform points in the camera frame to the pixel frame.
    Args:
        cam: pixel coordinates defined in the first camera coordinates system -- [B, 3, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] pixels coordinates if normalize is True else un-normalized coordinates -- [B, H, W, 2]
        mask: whether a pixel is on the image plane -- [B, 3, H, W]
    """
    b, _, h, w = cam.size()
    cam_flat = cam.reshape(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pixels = proj_c2p_rot.bmm(cam_flat)
    else:
        pixels = cam_flat

    if proj_c2p_tr is not None:
        pixels = pixels + proj_c2p_tr  # [B, 3, H*W]
    X = pixels[:, 0]
    Y = pixels[:, 1]
    Z = pixels[:, 2].clamp(min=1e-3)

    if normalize:
        X_norm = 2*((X / Z)/(w-1) - 0.5)  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
        Y_norm = 2*((Y / Z)/(h-1) - 0.5)  # Idem [B, H*W]
    else:
        X_norm = (X / Z)
        Y_norm = (Y / Z)
    X_mask = ((X_norm > 1) + (X_norm < -1)).detach()
    Y_mask = ((Y_norm > 1) + (Y_norm < -1)).detach()

    if padding_mode == 'zeros':
        X_norm[X_mask] = 2  # make sure that no point in warped image is a combination of im and gray
        Y_norm[Y_mask] = 2
    mask = ((X_norm > 1)+(X_norm < -1)+(Y_norm < -1)+(Y_norm > 1)).detach()
    mask = mask.unsqueeze(1).expand(b,3,h*w)

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    return pixel_coords.reshape(b,h,w,2), mask.reshape(b,3,h,w)


def warp(img, depth, pose_mat, intrinsics, padding_mode='border', inverse=True):
    global pixel_coords
    """
    Inverse warp a source image to the target image plane.

    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose_mat: 6DoF pose parameters from target to source as 4x4 matrix -- [B, 4, 4]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        Source image warped to the target image plane
        Flow in X direction [...,0] and Y direction [...,1] -- [H, W, 2]
        Mask indicating which pixels have been projected to target image -- [H, W, 1]
    """
    depth = depth.squeeze(1)

    check_sizes(depth, 'depth', 'BHW')
    check_sizes(intrinsics, 'intrinsics', 'B33')

    batch_size, _, img_height, img_width = img.size()

    # intrinsics = intrinsics.expand(img.shape[0],-1,-1)

    if (pixel_coords is None) or pixel_coords.size(2) < depth.size(1):
        pixel_coords = set_id_grid(depth)
    input = pixel_coords if inverse else torch.ones_like(depth).unsqueeze(1).expand((batch_size, 3, img_height, img_width))

    cam_points = pixel2cam(input, depth, torch.inverse(intrinsics))  # [B,3,H,W]
    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics.bmm((pose_mat if inverse else pose_mat.inverse())[:,:3,:])  # [B, 3, 4]
    rot, tr = proj_cam_to_src_pixel[:,:,:3], proj_cam_to_src_pixel[:,:,-1:]


    if inverse:
        src_pixel_coords, mask = cam2pixel(cam_points, rot, tr, padding_mode, normalize=True)  # [B,H,W,2]

        projected_img = F.grid_sample(img, src_pixel_coords, padding_mode=padding_mode, align_corners=False)
    else:
        src_pixel_flow, mask = cam2pixel(cam_points, rot, tr, padding_mode, normalize=False)  # [B,H,W,2]

        resample = Resample2d(bilinear=True)
        projected_img = resample(img, src_pixel_flow.transpose(1,3).contiguous())

        # fw = forward_warp()
        # projected_img = fw(img, src_pixel_flow)

    return projected_img, src_pixel_coords if inverse else src_pixel_flow, mask


def transform_code(z, RT, object_centric=False):
    b = z.size(0)

    z_tf = z.view(b,-1,3).bmm(RT[:,:3,:3])
    nz = z_tf.size(1)
    if not object_centric:
        z_tf = z_tf + RT[:,:3,3].unsqueeze(1).expand((-1,nz,3))
    return z_tf.view(b, nz * 3)