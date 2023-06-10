import torch


def psnr_metric(pred_rgb: torch.Tensor, gt_rgb: torch.Tensor):
    """Compute the Peak signal-to-noise ratio (PSNR) between the target and predictions.

    Args:
        pred_rgb (torch.Tensor): A tensor with the predicted RGB values.
        gt_rgb (torch.Tensor): A tensor with the target RGB values.
    """
    mse = torch.mean((pred_rgb - gt_rgb) ** 2)
    psnr = -10 * torch.log10(mse)
    return psnr.item()
