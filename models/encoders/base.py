from abc import ABC, abstractmethod
from typing import Dict, List
import torch
import torch.nn as nn


class EncoderBase(nn.Module, ABC):
    """
    Base class for all encoders (3D)

    -input : x -> tensor [Batch_size,Channels_in,Depth,Height,width]
    -Output : Dict[str,Tensor]
        {
        "stage1" : Tensor[:,:, D,H,W]
        "stage2" : Tensor[ :, :, D/2, H//2,W//2]
        ...

        }
        --- keys must be orderd from high resolution to low 
        --- Channel dimensions must match, self.feature_channels
        --- Downsampling must be consistent

    """

    def __init__(self, in_channels : int, feature_channels : List[int], ):
        super().__init__()
        self.in_channels = in_channels
        self.feature_channels = feature_channels
        self.num_stages = len(feature_channels)
        self._check_channels()
    
    def _check_channels(self):
        assert isinstance(self.feature_channels, list), \
            "feature_channels must be a list"
        assert all(isinstance(c, int) for c in self.feature_channels), \
            "feature_channels must be a list of ints"
        assert len(self.feature_channels) >= 2, \
            "Encoder should have at least 2 stages"
    
    @abstractmethod 
    def forward(self, x : torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Docstring for forward
        
        :param x: 
            Tensor [ B, in_channels , D , H , W]

        :returns:
            features: dict with keys:
                "stage1", "stage2", ...
        """
        raise NotImplementedError
    def _validate_outputs(
        self,
        features: Dict[str, torch.Tensor],
    ):
        """
        Optional safety check to be called in forward().
        Useful during development / debugging.

        """
        assert isinstance(features, dict), \
            "Encoder output must be a dict"

        assert len(features) == self.num_stages, \
            f"Expected {self.num_stages} stages, got {len(features)}"

        expected_keys = [f"stage{i+1}" for i in range(self.num_stages)]
        assert list(features.keys()) == expected_keys, \
        f"Expected keys {expected_keys}, got {list(features.keys())}"

        for i, (k, v) in enumerate(features.items()):
            assert isinstance(v, torch.Tensor), \
                f"Stage {k} is not a tensor"
            assert v.dim() == 5, \
                f"Stage {k} must be 5D [B,C,D,H,W], got {v.shape}"
            assert v.shape[1] == self.feature_channels[i], \
                f"Stage {k} channel mismatch: expected {self.feature_channels[i]}, got {v.shape[1]}"
