from typing import List
from torch import Tensor
import torch
import torch.nn as nn


class ResidualBlock_G(nn.Module):
    def __init__(self, C: int, lr: float):
        super(ResidualBlock_G, self).__init__()

        # params
        ## "residual blocks consisting of two convolutional layers with a kernel size five" from Scyclone paper
        kernel: int = 5

        # blocks
        self.conv_block = nn.Sequential(
            nn.Conv1d(C, C, kernel, padding=kernel // 2),
            nn.LeakyReLU(lr),
            nn.Conv1d(C, C, kernel, padding=kernel // 2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.conv_block(x)


class Generator(nn.Module):
    """
    Scyclone Generator
    """

    def __init__(self):
        super(Generator, self).__init__()

        # params
        n_C_freq: int = 128
        n_C_trunk: int = 256
        ## "In this study, we set nG and nD to 7 and 6, respectively" from Scyclone paper
        n_ResBlock_G: int = 7
        lr: float = 0.01

        # channel adjustment with pointwiseConv
        ## "We used leaky rectified linear units" from Scyclone paper
        blocks: List[nn.Module] = [
            nn.Conv1d(n_C_freq, n_C_trunk, 1),
            nn.LeakyReLU(lr),
        ]
        # Residual blocks
        blocks += [ResidualBlock_G(n_C_trunk, lr) for _ in range(n_ResBlock_G)]
        # channel adjustment with pointwiseConv
        blocks += [
            nn.Conv1d(n_C_trunk, n_C_freq, 1),
            nn.LeakyReLU(lr),
        ]

        # block registration (`*` is array unpack)
        self.model = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class ResidualSNBlock_D(nn.Module):
    def __init__(self, C: int, lr: float):
        super(ResidualSNBlock_D, self).__init__()

        # params
        ## "residual blocks consisting of two convolutional layers with a kernel size five" from Scyclone paper
        kernel = 5

        # blocks
        self.conv_blocks = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv1d(C, C, kernel, padding=kernel // 2)),
            nn.LeakyReLU(lr),
            nn.utils.spectral_norm(nn.Conv1d(C, C, kernel, padding=kernel // 2)),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Tensor[N, C, L] -> Tensor[N, C, L]
        """
        return x + self.conv_blocks(x)


class Discriminator(nn.Module):
    """
    Scyclone Discriminator
    """

    def __init__(self, n_batch: int = 64):
        super(Discriminator, self).__init__()

        # params
        n_C_freq: int = 128
        n_C_trunk: int = 256
        ## "In this study, we set nG and nD to 7 and 6, respectively" from Scyclone paper
        n_ResBlock_D: int = 6
        lr: float = 0.2

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # channel adjustment with pointwiseConv
        ## "We used leaky rectified linear units" from Scyclone paper
        blocks: List[nn.Module] = [
            nn.utils.spectral_norm(nn.Conv1d(n_C_freq, n_C_trunk, 1)),
            nn.LeakyReLU(lr),
        ]

        # Residual blocks
        blocks += [ResidualSNBlock_D(n_C_trunk, lr) for _ in range(n_ResBlock_D)]

        # final compression
        blocks += [
            nn.utils.spectral_norm(nn.Conv1d(n_C_trunk, 1, 1)),
            nn.LeakyReLU(lr),
            nn.AdaptiveAvgPool1d(1),
        ]

        # module registration (`*` is array unpack)
        self.model = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        # "We add small Gaussian noise following N (0, 0.01) to the input of the discriminator" from Scyclone paper
        return self.model(x + torch.randn(x.size().to(self._device)) * 0.01)
