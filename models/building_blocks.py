import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List, Optional

# Make sure you have monai installed: pip install monai

from affinity.self_fusegt3d import SelfFuseGT3D,MultiScaleSelfFuse3D

from affinity.affinity_utils3d import hard_affinity_threshold

from affinity.affinity_head_3d import AffinityHead3D

# Encoder,Decoder, UAFS, MAFS block, 

### ENCODER ####






    ### Always both streams will stay symmetric, Also I will  upsample symettrically to the encoders, so inchannels will be equal to the number of channels from encoder 
class UAFS(nn.Module):
        def __init__(self, in_channels : int ,out_channels : int,sc :int = 1):# sc means scale, Its not required to use varying scale in UAFS Blocks

            super().__init__()
            """
                args : 
                in_channels, out_channels [B,in_channels,D,H,W]
                returns :
                    segmentation features,[B,out_channels,D,H,W]
                    affinity features
            """
            self.sc = sc
            ## Upsampling
            self.up_s = nn.Sequential(
                    nn.ConvTranspose3d(
                        in_channels,
                        out_channels,
                        kernel_size=2,
                        stride=2,
                        bias=False
                    ),
                    nn.BatchNorm3d(out_channels),
                    nn.ReLU(inplace=True)
                )

            self.up_t = nn.Sequential(
                    nn.ConvTranspose3d(
                        in_channels,
                        out_channels,
                        kernel_size=2,
                        stride=2,
                        bias=False
                    ),
                    nn.BatchNorm3d(out_channels),
                    nn.ReLU(inplace=True)
            )

            ## PAD AFTER UPSCALING  NOT REQUIRED HERE

            ## this will take upscaled_s concatenated with residual features
            self.decoder_s = nn.Sequential(
                nn.Conv3d(out_channels+out_channels,out_channels,kernel_size=3,padding =1 ,stride =1,bias = False),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace = True),
                 nn.Conv3d(out_channels,out_channels,kernel_size=3,padding =1 ,stride =1,bias = False),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace = True)
            )
            ### tuning s to be concatenated with t
            self.s_to_t = nn.Sequential(
                nn.Conv3d(out_channels,out_channels,kernel_size = 1,stride =1,bias = False),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace = True)
            )
            ## This will take upscaled_t concatenated with refined features from s  
            self.decoder_t = nn.Sequential(
                nn.Conv3d(out_channels+out_channels,out_channels,kernel_size=3,padding =1 ,stride =1,bias = False),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace = True),
                nn.Conv3d(out_channels,out_channels,kernel_size=3,padding =1 ,stride =1,bias = False),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace = True)
            )


            ### to convert decoded t features to affinities
            self.inner_t = nn.Sequential(
                nn.Conv3d(out_channels,26,kernel_size = 3,padding = 1,stride = 1),
                ### Question --> shall i normalize here once
                nn.Sigmoid()
            )
            self.self_fusegt3d = SelfFuseGT3D(size = sc)


            ## Then we fuse things via self_fusegt3d 

            self.res_s = nn.Sequential(
                nn.Conv3d(out_channels,out_channels,kernel_size=3,padding = 1,stride = 1, bias= False),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace =True),
                nn.Conv3d(out_channels,out_channels,kernel_size=3,padding = 1,stride = 1, bias= False),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace =True),

            )


        ## now the result of this res_s(self_fusegt3d) will be merged with s via simple addition
        ## Question, why not some gated mechanism to add the 


        def forward(self,x_s,x_t,rf): 
            """                                    
                x_s --> segmentation branch input
                x_t --> affinity branch input
                rf --> skipped connection input
            """
            # print(x_s.shape)
            sc = self.sc
            x_s = self.up_s(x_s)
            x_t = self.up_t(x_t)

            ## PADDING Required ---> NOOO, because I will be using perfectly symmetric things, and input will be factor of two to the power depth

            assert rf.shape[2:]==x_s.shape[2:],\
                  "Residual shape is not same as x_s"
            if rf is not None: 
                x_s = torch.cat((x_s,rf),dim = 1)

            x_s = self.decoder_s(x_s)

            x_t = torch.cat([x_t, self.s_to_t(x_s) ],dim = 1)

            x_t = self.decoder_t(x_t)
            
            ## Affinities
            aff = self.inner_t(x_t)

            ## NOW PADDING IS ESSENTIAL --> CHECK INPUT INSTRUCTIONS OF SELF_FUSEGT3D

            t_cls = self.self_fusegt3d(F.pad(x_s,[self.sc,self.sc,self.sc,self.sc,self.sc,self.sc]),hard_affinity_threshold(aff))

            fuse_s = self.res_s(t_cls)
            
            x_s.add_(fuse_s)

            return x_s,x_t



class MAFS(nn.Module): 
    def __init__(self, in_channels: int, scales: List[int] = [1, 2, 3]):
        super().__init__()
        self.scales = scales

        # refineS refinement
        self.decoder_s = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
        )

        # s -> t
        self.s_to_t = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
        )

        # t decoder
        self.decoder_t = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
        )

        # Multi-scale affinity head
        self.affinity_head = AffinityHead3D(
            in_channels=in_channels,
            scales=scales,
            use_bn=False,
        )

        # Split heads per scale
        self.num_scales = len(scales)

        # Multi-scale fusion
        self.mssf = MultiScaleSelfFuse3D(
            scales=self.scales,
            learnable_weights=True,
        )

        # Final refinement
        self.res_s = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.output = nn.Sequential(
            nn.Conv3d(in_channels,in_channels//2,3,1,1,bias = False),
            nn.BatchNorm3d(in_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels//2, 1,3,1,1),
            nn.Sigmoid()
        )
    def forward(self, x_s: torch.Tensor, x_t: torch.Tensor): ## I MAKE A SKIPPED CONNECTION HERE
        # refine segmentation
        x_s = self.decoder_s(x_s)

        # fuse into topology stream
        t = torch.cat([x_t, self.s_to_t(x_s)], dim=1)
        x_t = self.decoder_t(t)

        # predict affinities
        aff_soft = self.affinity_head(x_t)  # [B, 26*K, D, H, W]

        hard_affs = [
            hard_affinity_threshold(aff_soft[:, k*26:(k+1)*26])
            for k in range(self.num_scales)
        ]

        fused_s = self.mssf(x_s, hard_affs)


        return self.output(x_s.add_(fused_s)), aff_soft

