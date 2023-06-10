from functools import partial
from typing import List, Optional, Union

import torch
from torch import nn


def activation_factory(name: str):
    return {
        "relu": nn.ReLU,
        "elu": partial(nn.ELU, alpha=1.0),
        "lrelu": partial(nn.LeakyReLU, negative_slope=0.1),
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "softplus": partial(nn.Softplus, beta=10.0),
    }[name]


def normalization_factory(name: str):
    return {
        "bn": partial(nn.BatchNorm1d, affine=True, eps=0.001),
        "in": partial(nn.InstanceNorm1d, affine=True, eps=0.001),
        "ln": partial(nn.LayerNorm, eps=1e-6),
    }[name]


def conv1d_block(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, List[int]],
    stride: int = 1,
    padding: int = 0,
    bias: bool = True,
    activation: str = None,
    normalization: str = None,
    momentum: float = 0.01,
) -> nn.Module:
    conv1d_module = nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
    )
    act = activation_factory(activation)() if activation else nn.Identity()
    norm = (
        normalization_factory(normalization)(out_channels, momentum=momentum)
        if normalization
        else nn.Identity()
    )
    return nn.Sequential(conv1d_module, norm, act)


def linear_block(
    in_channels: int,
    out_channels: int,
    bias: bool = True,
    activation: str = None,
    normalization: str = None,
    momentum: float = 0.01,
) -> nn.Module:
    bias = not normalization and bias
    linear_module = nn.Linear(
        in_features=in_channels,
        out_features=out_channels,
        bias=bias,
    )
    act = activation_factory(activation)() if activation else nn.Identity()
    norm = (
        normalization_factory(normalization)(out_channels, momentum=momentum)
        if normalization
        else nn.Identity()
    )
    return nn.Sequential(linear_module, norm, act)


def sample_pdf(
    bins: torch.Tensor,
    weights: torch.Tensor,
    N_samples: Optional[int] = 16,
    uniform: Optional[bool] = False,
):
    """This code is described in Section 3.3 and is inspired by
    https://github.com/yenchenlin/nerf-pytorch/blob/a15fd7cb363e93f933012fd1f1ad5395302f63a4/run_nerf_helpers.py#L196
    """
    weights = weights + 1e-5  # prevent NaNs
    # Compute the pdf along the ray
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    assert cdf.shape == weights.shape
    # Append 0.0 in every cdf
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    assert cdf.shape == bins.shape  # batch_size x n_rays x n_bins

    # Now we can implement the inverse sampling
    if uniform:
        u = torch.linspace(0.0, 1.0, steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])
    # Move everything to the correct device
    u = u.to(weights.device)

    # Invert CDF
    u = u.contiguous()
    # Now we need to locate in which bin of the CDF each one of the sampled u
    # falls in
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    # inds_g should have size batch_size x n_rays x N_samples x 2
    inds_g = torch.stack([below, above], -1)

    # Make the size of the cdf equal to batch_size x n_rays x N_samples x n_bins in
    # order to be able to do the gathering
    cdf_g = cdf.unsqueeze(2).expand(
        inds_g.shape[0],  # batch_size
        inds_g.shape[1],  # n_rays
        inds_g.shape[2],  # N_samples
        cdf.shape[-1],  # n_bins
    )
    # Gather the values from cdf_g along for each one of the N_samples
    cdf_g = torch.gather(cdf_g, dim=3, index=inds_g)

    bins_g = bins.unsqueeze(2).expand(
        inds_g.shape[0],  # batch_size
        inds_g.shape[1],  # n_rays
        inds_g.shape[2],  # N_samples
        cdf.shape[-1],  # n_bins
    )
    bins_g = torch.gather(bins_g, dim=3, index=inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples
