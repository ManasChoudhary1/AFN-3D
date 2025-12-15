import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention3d import SpatialAttention3D, ChannelAttention3D

from typing import Optional 

class GAU3D(nn.Module):
        """
        Gated Attention Unit (3D)

        Mathematical form:
            RF = M * x + (1 - M) * O

        where:
            - x : prior feature
            - O : candidate update
            - M : gate from spatial × channel attention

        Inputs:
            x : Tensor[B, C, D, H, W]
            y : Tensor[B, 1 or C, D, H, W]   (guidance signal)

        Output:
            RF: Tensor[B, C, D, H, W]
        """
        def __init__(
                    self,
                    in_channels : int,
                    use_gau : bool = True,
                    reduce_dim : bool = False,
                    out_channels : Optional[int] = None,
        ):
            super().__init__()
            self.use_gau = use_gau
            self.reduce_dim  = reduce_dim

            if self.reduce_dim:
                assert out_channels is not None, \
                  "out_channel must be set when reduce_dim = True"
                self.down_conv = nn.Sequential(
                      nn.Conv3d(in_channels,out_channels,kernel_size=1,bias = False),
                      nn.BatchNorm3d(out_channels),
                      nn.ReLU(inplace = True),
                )
                in_channels = out_channels
            
            if self.use_gau:
                 self.sa = SpatialAttention3D()
                 self.ca = ChannelAttention3D(in_channels)
                 self.reset_gate = nn.Sequential(
                      nn.Conv3d(
                           in_channels,
                           in_channels,
                           kernel_size= 3,
                           stride = 1,
                           padding = 2,
                           dilation = 2, ## TO increase the context range
                            bias = False
                      ),
                      nn.BatchNorm3d(in_channels),
                      nn.ReLU(inplace = True),
                 )
        def forward(self,x: torch.Tensor, y : torch.Tensor) -> torch.Tensor:
                """
                Args:
                    x: Tensor[B, C, D, H, W]   (prior feature)
                    y: Tensor[B, 1 or C, D, H, W] (guidance) should be in range [0,1]

                 Returns:
                    RF: refined feature Tensor[B, C, D, H, W]
                """
                if self.reduce_dim:
                     x = self.down_conv(x)
                if not self.use_gau:
                      return x
                if y.shape[2:] != x.shape[2:]:
                    y = F.interpolate(
                        y,
                        size=x.shape[2:],
                        mode="trilinear",
                        align_corners=True,
                    )
                comx = x*y   # how much x and y agree
                resx = x*(1-y) # how much they disagree

                x_sa = self.sa(resx)
                x_ca = self.ca(resx)

                O = self.reset_gate(comx)
                M = x_sa * x_ca

                RF = M*x + (1.0-M)*O  
                ## Refined Features
                ## This is like finding posterior (RF), given prior(x) and evidence (y)

                return RF                     