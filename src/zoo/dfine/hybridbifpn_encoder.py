"""  
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement  
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.  
---------------------------------------------------------------------------------  
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)  
Copyright (c) 2023 lyuwenyu. All Rights Reserved.  
"""  
  
import copy  
from collections import OrderedDict  
  
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
  
from ...core import register  
from .utils import get_activation  
from .hybrid_encoder import HybridEncoder  
from .bifpn_encoder import BiFPNEncoder, ConvNormLayer_fuse  
  
__all__ = ["HybridBiFPNEncoder"]  
  
  
@register()  
class HybridBiFPNEncoder(nn.Module):  
    __share__ = [  
        "eval_spatial_size",  
    ]  
  
    def __init__(  
        self,  
        in_channels=[512, 1024],  
        feat_strides=[16, 32],  
        hidden_dim=128,  
        use_encoder_idx=[1],  
        dim_feedforward=512,  
        expansion=0.34,  
        depth_mult=0.5,  
        num_bifpn_layers=2,  
        weighted_fusion=True,  
        use_attention=False,  
        attention_type=None,  
        dropout=0.0,  
        act="silu",  
        eval_spatial_size=None,  
    ):  
        super().__init__()  
          
        # Initialize BiFPN components  
        self.bifpn = BiFPNEncoder(  
            in_channels=in_channels,  
            feat_strides=feat_strides,  
            hidden_dim=hidden_dim,  
            num_bifpn_layers=num_bifpn_layers,  
            weighted_fusion=weighted_fusion,  
            use_attention=use_attention,  
            attention_type=attention_type,  
            dropout=dropout,  
            act=act,  
            eval_spatial_size=eval_spatial_size,  
        )  
          
        # Initialize Hybrid components  
        self.hybrid = HybridEncoder(  
            in_channels=in_channels,  
            feat_strides=feat_strides,  
            hidden_dim=hidden_dim,  
            use_encoder_idx=use_encoder_idx,  
            dim_feedforward=dim_feedforward,  
            expansion=expansion,  
            depth_mult=depth_mult,  
            eval_spatial_size=eval_spatial_size,  
        )  
          
        # Output properties - use the same as HybridEncoder for compatibility  
        self.out_channels = self.hybrid.out_channels  
        self.out_strides = self.hybrid.out_strides  
          
        # Store parameters for reference  
        self.in_channels = in_channels  
        self.feat_strides = feat_strides  
        self.hidden_dim = hidden_dim  
        self.eval_spatial_size = eval_spatial_size  
          
    def forward(self, feats):  
        # Process features through BiFPN first  
        bifpn_feats = self.bifpn.forward(feats)  
          
        # Then process through HybridEncoder  
        hybrid_feats = self.hybrid.forward(bifpn_feats)  
          
        return hybrid_feats