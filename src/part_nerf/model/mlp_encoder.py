from typing import List, Optional

import torch
from torch import nn


class MLPEncoder(nn.Module):
    """This class implements a multi-layer perceptron consisting of a sequence
    of fully connected layers followed by RELU non linearities.
    """

    def __init__(
        self,
        input_dims: int,
        proj_dims: List[int],
        non_linearity: nn.Module = nn.ReLU(),
        last_activation: nn.Module = nn.Identity(),
    ):
        """
        Args:
            input_dims (int): Number of channels of the input features.
            proj_dims (List[int]):List with the number of channels of each fully connected
                layer.
            non_linearity (nn.Module, optional): The non linear activation function.
                Defaults to nn.ReLU().
            last_activation (nn.Module, optional): Module implementing the activation
                after the last fully connected layer. Defaults to nn.Sequential().
        """
        super().__init__()
        # Make sure that a valid input is provided
        assert len(proj_dims) > 0
        self.last_activation = last_activation
        self.non_linearity = non_linearity

        # Append the input dims in the beginning
        proj_dims = [input_dims] + proj_dims

        # Start by appending the first fully connected layer
        self.layers = nn.ModuleList(
            [
                nn.Linear(in_dims, out_dims)
                for (in_dims, out_dims) in zip(proj_dims, proj_dims[1:])
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layers = self.layers[:-1]
        last_layer = self.layers[-1]

        for l in layers:
            x = self.non_linearity(l(x))

        return self.last_activation(last_layer(x))


class ResidualEncoder(nn.Module):
    """This class implements a residual encoder, used mostly for easier training of the
    NeRF variants."""

    def __init__(
        self,
        proj_dims: List[List[int]],
        out_dims: Optional[int] = -1,
        non_linearity: nn.Module = nn.ReLU(),
        last_activation: nn.Module = nn.Identity(),
    ):
        """
        Args:
            proj_dims (List[List[int]]): List of lists with the channels configuration for
                each residual block.
            out_dims (Optional[int]): The output dim for the residual encoder.
            non_linearity (nn.Module, optional): The non linear activation function.
                Defaults to nn.ReLU().
            last_activation (nn.Module, optional): The non linearity used in the end.
                Defaults to nn.Identity().
        """
        super().__init__()
        # Make sure that a valid input is provided
        assert len(proj_dims) >= 1
        self.last_activation = last_activation
        self.non_linearity = non_linearity
        self.n_blocks = len(proj_dims)

        self.residual_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    proj_dims[i][0],
                    proj_dims[i][1],
                    proj_dims[i][2],
                    non_linearity=non_linearity,
                )
                for i in range(self.n_blocks)
            ]
        )
        if out_dims == -1:
            out_dims = proj_dims[-1][-1]
        self.fc_out = nn.Linear(proj_dims[-1][-1], out_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.residual_blocks:
            x = block(x)
        return self.last_activation(self.fc_out(x))


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_dims: int,
        proj_dims: int,
        out_dims: Optional[int] = None,
        non_linearity: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        if out_dims is None:
            out_dims = in_dims

        if in_dims == out_dims:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(in_dims, out_dims, bias=False)

        self.block = nn.Sequential(
            non_linearity,
            nn.Linear(in_dims, proj_dims),
            non_linearity,
            nn.Linear(proj_dims, out_dims),
        )

        # Initialization
        nn.init.zeros_(self.block[-1].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dx = self.block(x)
        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x
        return x_s + dx
