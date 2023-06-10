import torch
from torch import nn
from torch.nn import functional as F


class CBatchNorm1d(nn.Module):
    """Conditional batch normalization layer class.
    Adapted from https://github.com/autonomousvision/occupancy_networks.

    Args:
        c_dim (int): The dimension of latent conditioned code c.
        f_dim (int): The feature dimension.
        norm_method (str, optional): The normalization method. Defaults to None.
    """

    def __init__(self, c_dim: int, f_dim: int, norm_method: str = None):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.norm_method = norm_method
        # Submodules
        self.conv_gamma = nn.Conv1d(c_dim, f_dim, 1)
        self.conv_beta = nn.Conv1d(c_dim, f_dim, 1)
        if norm_method is None:
            self.bn = nn.Identity()
        elif norm_method == "bn":
            self.bn = nn.BatchNorm1d(f_dim, affine=False)
        elif norm_method == "in":
            self.bn = nn.InstanceNorm1d(f_dim, affine=False)
        else:
            raise ValueError("Invalid normalization method!")
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.conv_gamma.weight)
        nn.init.zeros_(self.conv_beta.weight)
        nn.init.ones_(self.conv_gamma.bias)
        nn.init.zeros_(self.conv_beta.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        assert x.size(0) == c.size(0)
        assert c.size(1) == self.c_dim

        # c is assumed to be of size batch_size x c_dim x T
        if len(c.size()) == 2:
            c = c.unsqueeze(2)

        # Affine mapping
        gamma = self.conv_gamma(c)
        beta = self.conv_beta(c)

        # Batchnorm
        net = self.bn(x)
        out = gamma * net + beta

        return out


class CResnetBlockConv1d(nn.Module):
    """Conditional batch normalization-based Resnet block class.
    Adapted from https://github.com/autonomousvision/occupancy_networks.

    Args:
        c_dim (int): dimension of latend conditioned code c
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
        norm_method (str): normalization method
    """

    def __init__(
        self,
        c_dim: int,
        size_in: int,
        size_h: int = None,
        size_out: int = None,
        norm_method: str = None,
    ):
        super().__init__()
        # Attributes
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.bn_0 = CBatchNorm1d(c_dim, size_in, norm_method=norm_method)
        self.bn_1 = CBatchNorm1d(c_dim, size_h, norm_method=norm_method)

        self.fc_0 = nn.Conv1d(size_in, size_h, 1)
        self.fc_1 = nn.Conv1d(size_h, size_out, 1)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        net = self.fc_0(self.actvn(self.bn_0(x, c)))
        dx = self.fc_1(self.actvn(self.bn_1(net, c)))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class MiniOccupancyNet(nn.Module):
    """Occupancy Network Decoder using Conditional Batch Normalization.
    Adapted from https://github.com/autonomousvision/occupancy_networks.

    Args:
        dim (int): The input dimension. Defaults to 3.
        c_dim (int): The dimension of the latent conditioned code c. Defaults to 128.
        hidden_size (int): The hidden size of the Occupancy network. Defaults to 256.
        n_blocks (int): The number of ResNet blocks. Defaults to 1
        norm_method (str, optional): The normalization method. Defaults to None.
        with_sigmoid (bool): Flag denoting if we apply sigmoid in the final output. Defaults to False.
    """

    def __init__(
        self,
        dim: int = 3,
        c_dim: int = 128,
        out_dim: int = 1,
        hidden_size: int = 256,
        n_blocks: int = 1,
        norm_method: str = None,
        with_sigmoid: bool = False,
    ):
        super().__init__()
        self._dim = dim
        self._with_sigmoid = with_sigmoid
        self._out_dim = out_dim
        self.conv_p = nn.Conv1d(dim, hidden_size, 1)
        self.blocks = nn.ModuleList(
            [
                CResnetBlockConv1d(c_dim, hidden_size, norm_method=norm_method)
                for _ in range(n_blocks)
            ]
        )

        self.bn = CBatchNorm1d(c_dim, hidden_size, norm_method=norm_method)
        self.conv_out = nn.Conv1d(hidden_size, out_dim, 1)
        self.actvn = nn.ReLU()

    def forward(self, p: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        C = p.shape[-1]
        assert C == self._dim
        p = p.transpose(1, 2)

        net = self.conv_p(p)

        for block in self.blocks:
            net = block(net, c)

        out = self.conv_out(self.actvn(self.bn(net, c)))
        out = out.transpose(1, 2)
        if self._out_dim == 1 and self._with_sigmoid:
            out = F.sigmoid(out)
        return out
