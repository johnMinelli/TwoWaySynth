from __future__ import division

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from models_skip.network_utils.metrics import ssim
from models_skip.network_utils.projection_layer import inverse_warp


# Depth loss took and adapted from SfMLearner - "Unsupervised Learning of Depth and Ego-Motion from Video"
def photometric_reconstruction_loss(tgt_img, ref_img, intrinsics, depth_scales, pose):
    def one_scale(depth):

        b, _, h, w = depth.size()
        downscale = tgt_img.size(2)/h

        # rescale and normalize images which are in [-1,1]
        tgt_img_scaled = F.interpolate(tgt_img * 0.5 + 0.5, (h, w), mode='area')
        ref_img_scaled = F.interpolate(ref_img * 0.5 + 0.5, (h, w), mode='area')
        intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)

        ref_img_warped, _, valid_points = inverse_warp(ref_img_scaled, depth[:,0], pose, intrinsics_scaled)

        # consider the difference between (only) projected points respect to the target image
        diff = ((tgt_img_scaled - ref_img_warped) * (1-valid_points.float()))
        # mask the background (black)
        reconstruction_loss = diff[tgt_img_scaled>0].abs().mean()

        return reconstruction_loss, ref_img_warped, diff

    warped_results, diff_results = [], []
    if type(depth_scales) not in [list, tuple]:
        depth_scales = [depth_scales]

    total_loss = 0
    for d in depth_scales:
        loss, warped, diff = one_scale(d)
        total_loss += loss
        warped_results.append(warped)
        diff_results.append(diff)
    return total_loss, warped_results, diff_results

def get_depth_smoothness(depth, img):
    d_gradients_x = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])
    d_gradients_y = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:])

    image_gradients_x = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])
    image_gradients_y = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])

    weights_x = torch.exp(-10.0 * torch.mean(image_gradients_x, 1, keepdim=True))
    weights_y = torch.exp(-10.0 * torch.mean(image_gradients_y, 1, keepdim=True))

    smoothness_x = torch.mean(d_gradients_x * weights_x)
    smoothness_y = torch.mean(d_gradients_y * weights_y)

    return smoothness_x + smoothness_y


def depth_loss(d_true, d_pred):

    w1, w2, w3 = 1.0, 1.0, 0.1

    mask = [d_true>0]
    l_depth = torch.mean(torch.abs(d_pred[mask] - d_true[mask]), axis=-1)

    dx_pred = torch.abs(d_pred[:, :, :-1, :] - d_pred[:, :, 1:, :])
    dy_pred = torch.abs(d_pred[:, :, :, :-1] - d_pred[:, :, :, 1:])
    dx_true = torch.abs(d_pred[:, :, :-1, :] - d_pred[:, :, 1:, :])
    dy_true = torch.abs(d_pred[:, :, :, :-1] - d_pred[:, :, :, 1:])

    l_edges = torch.mean(torch.abs(dy_pred - dy_true) + torch.abs(dx_pred - dx_true), axis=-1)

    l_ssim = torch.clip((1 - ssim(d_true, d_pred)) * 0.5, 0, 1)

    return (w1 * l_ssim) + (w2 * torch.mean(l_edges)) + (w3 * torch.mean(l_depth))

