import torch
import torch.nn as nn
from typing import List, Optional





class AffinityHead3D(nn.Module):
    """
    Affinity prediction head for 3D AFN-style models.

    Supports:
        - Single-scale affinity (UAFS equivalent): 26 directions
        - Multi-scale affinity (MAFS equivalent): 26 * num_scales directions

    This module ONLY predicts affinities.
    No thresholding, no fusion, no topology logic here.
    """



    def __init__(
            self,
            in_channels : int,
            scales : Optional[List[int]] = None,
            dilation : int  = 1,
            use_bn : bool = False
                  ):
        super().__init__()
        """
            Args:
                in_channels: number of input feature channels
                scales:
                    - None or [1]  -> single-scale (26)
                    - [1, 2, 3]    -> multi-scale (26 * 3)
                dilation: dilation for affinity prediction conv
                use_bn: whether to use BatchNorm3d
        """
        if scales is None:
            scales = [1]
            
        self.scales = scales
        self. num_scales = len(scales)
        self.num_dirs = 26
        self.out_channels = self.num_dirs * self.num_scales

        padding = dilation

        layers = [
            nn.Conv3d(
                in_channels,
                self.out_channels,
                kernel_size=3,
                padding = padding,
                dilation = dilation,
                bias = not use_bn,
            )
        ]

        if use_bn :
            layers.append(nn.BatchNorm3d(self.out_channels))

        self.conv = nn.Sequential(*layers)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor[B, C, D, H, W]

        Returns:
            affinity_soft:
                - Single-scale: [B, 26, D, H, W]
                - Multi-scale:  [B, 26 * num_scales, D, H, W]
        """
        return self.act(self.conv(x))
