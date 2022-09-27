import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel, sigma=0.5):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True, mask=None):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        if mask is None:
            return ssim_map.mean(1).mean(1).mean(1)
        else:
            return (ssim_map.mean(1)[mask]).mean()


def ssim(img1, img2, window_size=11, size_average=True, mask=None, sigma=0.5):
    img1 = img1.mean(1).unsqueeze(1)
    img2 = img2.mean(1).unsqueeze(1)

    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel, sigma)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average, mask=mask)


# Metrics from monodepth2 https://arxiv.org/pdf/1806.01260.pdf "Digging Into Self-Supervised Monocular Depth Estimation"
def _compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())
    # Relative absolute error (percent)
    abs_rel = torch.mean(torch.abs(gt - pred) / gt)
    # Relative squared error (percent)
    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    err = torch.log(pred) - torch.log(gt)
    # Scale invariant logarithmic error [log(m)*100]
    silog = torch.sqrt(torch.mean(err ** 2) - torch.mean(err) ** 2) * 100
    if torch.isnan(silog):
        silog = torch.tensor(0)

    log_10 = (torch.abs(torch.log10(gt) - torch.log10(pred))).mean()

    return abs_rel, sq_rel, rmse, log_10, rmse_log, silog, a1, a2, a3


def compute_depth_metrics(depth_gt, depth_pred, min_depth=1e-3, max_depth=80):
    """Compute depth metrics, to allow monitoring during training

    This isn't particularly accurate as it averages over the entire batch,
    so is only used to give an indication of validation performance
    """
    # depth_pred = torch.clamp(F.interpolate(depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
    depth_pred = depth_pred.detach()

    mask_1 = depth_gt > min_depth
    mask_2 = depth_gt < max_depth
    mask = torch.logical_and(mask_1, mask_2)

    depth_gt = depth_gt[mask]
    depth_pred = depth_pred[mask]
    depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)
    depth_pred = torch.clamp(depth_pred, min=min_depth, max=max_depth)

    depth_errors = _compute_depth_errors(depth_gt, depth_pred)

    depth_metric_names = ["depth_abs_rel", "depth_sq_rel", "depth_rms", "depth_log10", "depth_log_rms", "depth_silog", "depth_a1", "depth_a2", "depth_a3"]

    metrics = {}
    for i, metric_name in enumerate(depth_metric_names):
        metrics[metric_name] = depth_errors[i].item()
    return metrics