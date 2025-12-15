import torch
from torch.nn import nn
import torch.nn.functional as F

from typing import Optional, Tuple, Callable, List

from .affinity_utils3d import get_groupfeature_3d
## This module not only extrapolates the 2d logic of AFN to 3d, 
## but also in a robust manner so as to not get into OOM errors


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
                    ]

                    aff = affinity[:, idx : idx + 1]

                    # context += neigh * aff   (NO TEMP TENSORS)
                    torch.addcmul_(context, neigh, aff)

                    idx += 1

        return context




## THIS RETURN THE UPDATED FEATURE
class MultiScaleSelfFuse3D(nn.Module):
    """
    Memory-optimal multi-scale affinity fusion (3D).

    - Zero-copy neighborhood access
    - In-place arithmetic
    - No redundant module instances
    """

    def __init__(
        self,
        num_scales: int,
        learnable_weights: bool = False,
    ):
        super().__init__()

        assert num_scales >= 1
        self.num_scales = num_scales

        # Single instance is sufficient (stateless)

        if learnable_weights:
            self.scale_weights = nn.Parameter(torch.zeros(num_scales))
        else:
            self.register_buffer("scale_weights", torch.zeros(num_scales))

    def forward(
        self,
        x: torch.Tensor,
        affinity_heads,
        sizes,
    ) -> torch.Tensor:
        """
        Args:
            x: Tensor[B, C, D, H, W]
            affinity_heads: list of callables (x -> [B,26,D,H,W])
            sizes: list of neighborhood radii (len = num_scales)

        Returns:
            fused feature: Tensor[B, C, D, H, W]
        """
        weights = torch.softmax(self.scale_weights, dim=0)
        fused = x

        for k in range(self.num_scales):
            s = sizes[k]

            # Pad once per scale
            padded_x = F.pad(
                x,
                (s, s, s, s, s, s),
                mode="constant",
                value=0,
            )

            affinity = affinity_heads[k](x)

            # Compute context
            context = SelfFuseGT3D(s)(
                padded_x,
                affinity
            )

            # In-place scale and accumulate
            context.mul_(weights[k])
            fused.add_(context)

            # Explicit deletion to save memory
            del padded_x, affinity, context

        return fused
