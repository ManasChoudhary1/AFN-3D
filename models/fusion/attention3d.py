import torch
import torch.nn as nn


class SpatialAttention3D(nn.Module):
    """
    Spatial attention for 3D feature maps.

    Input:
        x: Tensor[B, C, D, H, W]

    Output:
        attention map: Tensor[B, 1, D, H, W]
    """
     
    def __init__(self):
          super().__init__()
          self.conv = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.Conv3d(1, 1, kernel_size=5, padding=2, bias=False),
            nn.Sigmoid(),
          )
    

    def forward(self,x : torch.Tensor) -> torch.Tensor:
     ## Channel wise average and max
        avg_out = torch.mean(x,dim = 1,keepdim=True) # [B,1,...]
        max_out, _ = torch.max(x,dim = 1,keepdim=True) # B,1, ...

        x = torch.cat([avg_out,max_out],dim = 1) # B,2, ...

        return self.conv(x) ## spatial attention

class ChannelAttention3D(nn.Module):
    
    """
    Channel attention for 3D feature maps.

    Input:
        x: Tensor[B, C, D, H, W]

    Output:
        attention map: Tensor[B, C, 1, 1, 1] 
     Gives Channel wise attention contrary to Spatial attention
    """
    def __init__(self, channels : int, reduction = 2):
        super().__init__()

        assert channels% reduction == 0,\
            "channels must be divisible by reduction"
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc1 = nn.Conv3d(channels, channels // reduction,kernel_size=1,bias = False)
        #self.relu = nn.ReLU(inplace = True) ## Ommitting ---> Like the AFS paper
        self.fc2 = nn.Conv3d(channels//reduction,channels,kernel_size = 1,bias = False)

        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc2(self.fc1(self.avg_pool(x)))
        max_out =  self.fc2(self.fc1(self.max_pool(x)))
        return self.act(avg_out + max_out)

