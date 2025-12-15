import torch
import torch.nn as nn
from typing import Dict

from .base import EncoderBase

### Just a demo baseline


class ConvEncoder3D(EncoderBase):
    """
    Simple 3D convolutional encoder.

    Downsampling rule:
    - stage1: no downsampling
    - each next stage: downsample by factor 2 using stride-2 conv

    Output:
        {
            "stage1": [B, C1, D,   H,   W  ],
            "stage2": [B, C2, D/2, H/2, W/2],
            "stage3": [B, C3, D/4, H/4, W/4],
            ...
        }
    """
    def __init__(self, in_channels : int, feature_channels : list[int]):
        super().__init__(in_channels,feature_channels)

        self.stages = nn.ModuleList()

        prev_channels = in_channels
        for i, out_channels in enumerate(feature_channels):
            if i==0 :
                stage = self._make_stage(
                    prev_channels,
                    out_channels,
                    stride = 1,
                )
            else :
                stage = self._make_stage(
                    prev_channels,
                    out_channels,
                    stride = 2
                )
            self.stages.append(stage)
            prev_channels = out_channels
    def _make_stage(
                self,
                in_channels : int,
                out_channels: int,
                stride : int,
        )-> nn.Sequential:
            """
            Conv3D --> BatchNorm --> ReLU            
            """
            return nn.Sequential(
            # Layer 1: Downsample (or keep size)
            nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            
            # Layer 2: Refine features (Keep size constant)
            nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x: torch.Tensor)-> Dict[str,torch.Tensor]:
            """
            Args : 
            x : Tensor[ B, C_in, D, H, W]

            Returns:
                features: dict with keys "stage1", "stage2",..
            
            """
            features = {}
            for i, stage in enumerate(self.stages):
                x = stage(x)
                features[f"stage{i+1}"] = x

            return features