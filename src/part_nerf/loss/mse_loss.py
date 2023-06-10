import torch


def mse_loss(predictions: torch.Tensor, targets: torch.Tensor):
    return torch.mean((predictions - targets) ** 2)


def mse_loss_positive(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    no_rendering_rays_mask: torch.Tensor,
):
    rendering_rays_mask = ~no_rendering_rays_mask  # (B, N)
    color_dist = (predictions - targets) ** 2  # (B, N, 3)
    vector_dim = color_dist.shape[-1]
    num_positive_rays = torch.sum(rendering_rays_mask)
    positive_ray_preds = color_dist * rendering_rays_mask[..., None]
    summed_loss = torch.sum(positive_ray_preds)
    if num_positive_rays == 0:
        loss = summed_loss * 0.0
    else:
        loss = summed_loss / (vector_dim * num_positive_rays)
    return loss
