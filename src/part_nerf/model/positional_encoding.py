import math

import torch


class FixedPositionalEncoding(torch.nn.Module):
    """Implements the Fixed Positional Encoding function
    \gamma(\bx) = [
        \bx,
        \sin(2^0\pi\bx),
        \cos(2^0\pi\bx),
        ...,
        \sin(2^(proj_dims-1)\pi\bx),
        \cos(2^(proj_dims-1)\pi\bx)
    ]
    as defined in Eq. (4) in the original NeRF paper.
    """

    def __init__(self, proj_dims: int):
        super().__init__()
        ll = proj_dims // 2
        exb = torch.linspace(0, ll - 1, ll)
        self.register_buffer("sigma", math.pi * torch.pow(2.0, exb).view(1, -1))

    def forward(self, x):
        """
        Arguments:
        ----------
            x: A torch.tensor of size ...xC containing a tensor with C
               dimensions to be mapped into a higher dimensional space using a
               fixed positional encoding. C can be 1 or any other positive
               number.
        """
        x_shape = tuple(x.shape[:-1])
        if x.shape[-1] != 1:
            x = x.unsqueeze(-1)

        return torch.cat(
            [x, torch.sin(x * self.sigma), torch.cos(x * self.sigma)], dim=-1
        ).reshape(x_shape + (-1,))
