"""
Software Name : QASTAnet
SPDX-FileCopyrightText: Copyright (c) Orange SA
SPDX-License-Identifier: MIT
This software is distributed under the MIT License,
see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html
Authors: Adrien Llave, Grégory Pallone
Software description: An objective metric assessing the global audio quality of 3D audio signals (binaural or (higher-order) ambisonics)
"""

import torch
from torch import nn
import torch.nn.functional as F


class Max(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        max, max_indices = torch.max(x, dim=self.dim)
        return max


class SWAP(nn.Module):
    """
    Softmax-weighted Average Pooling (SWAP)
    See:
        - http://towardsdatascience.com/swap-softmax-weighted-average-pooling-70977a69791b/
        - Boureau et al. 2010
        - Gao et al. 2019

    """

    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
        self.beta = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        weights = F.softmax(self.beta * x, dim=self.dim)
        x = torch.einsum("...j, ...i -> ...", weights, x)

        return x


class PNorm(nn.Module):
    """
    Powered norm
    See: Boureau et al. 2010

    """

    def __init__(self, p=2.0, dim=-1):
        super().__init__()
        self.dim = dim
        self.p = torch.nn.Parameter(torch.tensor(p))
        self.eps = 1e-6

    def forward(self, x):
        """
        x is assumed to be > 0
        """
        return torch.mean(torch.abs(x) ** self.p + self.eps, dim=self.dim) ** (
            1 / self.p
        )
