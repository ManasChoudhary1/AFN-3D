import torch
import torch.nn.functional as F
from typing import List



def get_groupfeature_3d(
        feature : torch.Tensor,
        size : int,
    
) -> List[torch.Tensor]: ## Features from multiple directions
    """
    Collect 26-neighborhood shifted features in 3D.

    Args:
        feature: Tensor[B, C, D, H, W] (should already be  padded)
        size: shift radius (usually 1)

    Returns:
        neighbors: list of 26 tensors, each [B, C, D, H, W]
                   order is fixed and consistent
    """
    assert feature.dim() == 5, "feature must be 5D [B,C,D,H,W]"
    s = size

    neighbors = []

    for dz in (-s,0,s):
        for dy in (-s,0,s):
            for dx in (-s,0,s):
                if dz==0 and dy==0 and dx==0:
                    continue # removing the center one

                z_slice = slice(s + dz, -s + dz if -s + dz != 0 else None)
                y_slice = slice(s + dy, -s + dy if -s + dy != 0 else None)
                x_slice = slice(s + dx, -s + dx if -s + dx != 0 else None)

                neigh = feature[:, :, z_slice, y_slice, x_slice].clone()
                neighbors.append(neigh)

    assert len(neighbors) == 26, f"Expected 26 neighbors, got {len(neighbors)}"
    return neighbors  




def hard_affinity_threshold(
        affinity: torch.Tensor,
) -> torch.Tensor:
    """
    AFN-style hard thresholding. 
    It is a point to note that, I considered applying soft thresholds which you too might think about,
    but think of affinity as learning stuff from voxels that are actually from the same class,
    Soft thresholds leads to smoothing, and also thick boundaries, which we don't want.
    If someone wants to try soft thresholds, I would recommend tuning the temperature coefficient such that it still is hard enough.
    
    
    
    For each voxel, compute mean over directions.
    affinity > mean → 1
    else → 0

    Args:
        affinity: Tensor[B, K, D, H, W]
                  K = number of directions (26 or multiples)

    Returns:
        binary_affinity: Tensor[B, K, D, H, W] ∈ {0,1}
    """
    assert affinity.dim() == 5, "affinity must be [B,K,D,H,W]"

    mean_aff = affinity.mean(dim=1, keepdim=True)
    binary = (affinity > mean_aff).float()
    return binary
   



def make_gt_to_affinity_3d(
    gt: torch.Tensor,
    size: int,
) -> torch.Tensor:
    """
    Convert 3D ground-truth labels to affinity maps.

    Args:
        gt: Tensor[B, 1, D, H, W], binary ground truth
        size: neighborhood radius (usually 1)

    Returns:
        affinity_gt: Tensor[B, 26, D, H, W] ∈ {0,1} # This means 0 or one, not 0 to one
    """
    assert gt.dim() == 5, "gt must be [B,1,D,H,W]"
    assert gt.size(1) == 1, "gt must have a single channel"

    s = size

    # Pad gt so neighbor slices are valid
    gt_pad = F.pad(
        gt,
        (s, s, s, s, s, s),  # (Wl, Wr, Hl, Hr, Dl, Dr)
        mode="constant",
        value=0,
    )

    affinities = []

    for dz in (-s, 0, s):
        for dy in (-s, 0, s):
            for dx in (-s, 0, s):
                if dz == 0 and dy == 0 and dx == 0:
                    continue

                z_slice = slice(s + dz, -s + dz if -s + dz != 0 else None)
                y_slice = slice(s + dy, -s + dy if -s + dy != 0 else None)
                x_slice = slice(s + dx, -s + dx if -s + dx != 0 else None)

                gt_neigh = gt_pad[:, :, z_slice, y_slice, x_slice]

                # affinity = NOT XOR
                aff = torch.logical_not(
                    torch.logical_xor(gt.bool(), gt_neigh.bool())
                )

                affinities.append(aff.float())

    affinity_gt = torch.cat(affinities, dim=1)
    assert affinity_gt.shape[1] == 26, \
        f"Expected 26 affinity channels, got {affinity_gt.shape[1]}"

    return affinity_gt

# install scikit-image libary
from scipy.ndimage import label as cc_label


def make_gt_to_affinity_3d_cc(
    gt: torch.Tensor,
    size: int,
) -> torch.Tensor:
    """
    CC-aware 3D affinity ground truth for Vesuvius.

    Args:
        gt:   Tensor [B, 1, D, H, W] with values {0=bg, 1=scroll, 2=ignore}
        size: neighborhood radius (1,2,3)

    Returns:
        affinity_gt: Tensor [B, 26, D, H, W] in {0,1}
    """
    assert gt.dim() == 5 and gt.size(1) == 1

    B, _, D, H, W = gt.shape
    s = size
    device = gt.device

    # pad GT
    gt_pad = F.pad(
        gt,
        (s, s, s, s, s, s),
        mode="constant",
        value=0,
    )

    # ---------- compute CC maps (CPU, once per sample) ----------
    cc_maps = []
    for b in range(B):
        gt_np = gt[b, 0].cpu().numpy()
        scroll = (gt_np == 1)
        cc, _ = cc_label(scroll)  # 0 = background # type: ignore
        cc_maps.append(torch.from_numpy(cc))

    cc_maps = torch.stack(cc_maps, dim=0).to(device)  # [B,D,H,W]
    cc_pad = F.pad(
        cc_maps.unsqueeze(1),
        (s, s, s, s, s, s),
        mode="constant",
        value=0,
    )[:, 0]

    affinities = []

    for dz in (-s, 0, s):
        for dy in (-s, 0, s):
            for dx in (-s, 0, s):
                if dz == dy == dx == 0:
                    continue

                z = slice(s + dz, -s + dz if -s + dz != 0 else None)
                y = slice(s + dy, -s + dy if -s + dy != 0 else None)
                x = slice(s + dx, -s + dx if -s + dx != 0 else None)

                c = gt[:, 0]
                n = gt_pad[:, 0, z, y, x]

                cc_c = cc_maps
                cc_n = cc_pad[:, z, y, x]

                # valid voxels (ignore handled later by mask)
                valid = (c != 2) & (n != 2)

                # bg–bg
                bb = (c == 0) & (n == 0)

                # scroll–scroll same CC
                ss_same = (c == 1) & (n == 1) & (cc_c == cc_n)

                aff = (bb | ss_same) & valid
                affinities.append(aff.unsqueeze(1).float())

    affinity_gt = torch.cat(affinities, dim=1)
    assert affinity_gt.shape[1] == 26

    return affinity_gt
