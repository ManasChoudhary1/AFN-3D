import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Callable, List

from .affinity_utils3d import get_groupfeature_3d


## REMEMBER THIS CLASS DOES NOT RETURN UPDATED FEATURES BUT THE CONTEXT ( UPDATED FESTURES - X)
class SelfFuseGT3D(nn.Module):
    """
    Single-scale affinity-guided context aggregation (3D).

    Returns ONLY the aggregated context:
        context = sum_d (aff_d * neighbor_d)
    """

    def __init__(self,size:int =1 ):
        super().__init__()
        self.size = size
    def forward(
        self,
        padded_feature: torch.Tensor,
        affinity: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
        ## IMPORTANT ##
            "padded_feature": Tensor[B, C, D+2s, H+2s, W+2s]
            affinity: Tensor[B, 26, D, H, W]
            size: neighborhood radius

        Returns:
            context: Tensor[B, C, D, H, W]
        """
        B, C = padded_feature.shape[:2]
        D, H, W = affinity.shape[2:]

        # allocate context ONCE
        context = torch.zeros(
            (B, C, D, H, W),
            device=padded_feature.device,
            dtype=padded_feature.dtype,
        )

        idx = 0
        s = self.size

        for dz in (-s, 0, s):
            for dy in (-s, 0, s):
                for dx in (-s, 0, s):
                    if dz == 0 and dy == 0 and dx == 0:
                        continue

                    neigh = padded_feature[
                        :,
                        :,
                        s + dz : s + dz + D,
                        s + dy : s + dy + H,
                        s + dx : s + dx + W,
                    ].clone()

                    aff = affinity[:, idx : idx + 1]

                    # context += neigh * aff   (NO TEMP TENSORS)
                    context = context +  neigh *aff
                    idx += 1

        return context




## THIS RETURN THE UPDATED FEATURE, ALSO I DONT USE WEIGHTS DEPENDENT ON FEATURES, SIMPLY BECAUSE I DON"T FEEL THE TASK I AM MAKING THIS FOR REQUIRES IT
class MultiScaleSelfFuse3D(nn.Module):
    def __init__(self, scales: List[int], learnable_weights: bool = False):
        super().__init__()
        self.scales = scales
        self.num_scales = len(scales)

        if learnable_weights:
            self.scale_weights = nn.Parameter(torch.zeros(self.num_scales))
        else:
            self.register_buffer("scale_weights", torch.zeros(self.num_scales))

        self.fusers = nn.ModuleList(
            [SelfFuseGT3D(size=s) for s in scales]
        )

    def forward(
        self,
        x: torch.Tensor,
        hard_affinities: List[torch.Tensor],  # ⬅️ IMPORTANT
    ) -> torch.Tensor:
        """
        Args:
            x: [B, C, D, H, W]
            hard_affinities: list of [B, 26, D, H, W] (binary)

        Returns:
            fused feature: [B, C, D, H, W]
        """
        weights = torch.softmax(self.scale_weights, dim=0)
        fused = x.clone()
        # X is consumed here, assuming no further use of this x

        for k, (s, aff) in enumerate(zip(self.scales, hard_affinities)):
            padded_x = F.pad(x, (s, s, s, s, s, s))
            context = self.fusers[k](padded_x, aff)
            fused = fused + (weights[k] * context)

        return fused
