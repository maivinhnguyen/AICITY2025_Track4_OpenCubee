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
from .hybrid_encoder import (  
    TransformerEncoder,  
    TransformerEncoderLayer,  
    ConvNormLayer_fuse,  
    SCDown,  
    RepNCSPELAN4  
)  
  
__all__ = ["HybridBiFPNEncoder"]  
  
  
class WeightedFeatureFusion(nn.Module):  
    """  
    Weighted feature fusion as used in BiFPN.  
    Learns weights for each input feature map and performs weighted sum.  
    """  
    def __init__(self, num_inputs):  
        super().__init__()  
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32), requires_grad=True)  
          
    def forward(self, *inputs):  # Accept variable arguments instead of a list  
        weights = F.softmax(self.weights, dim=0)  
        return sum(w * x for w, x in zip(weights, inputs))  
  
  
class BiFPNBlock(nn.Module):  
    """  
    Single BiFPN block with bidirectional cross-scale connections  
    """  
    def __init__(self, hidden_dim, expansion=1.0, depth_mult=1.0):  
        super().__init__()  
          
        # Top-down pathway (from higher level to lower level)  
        self.td_convs = nn.ModuleList()  
        self.td_weights = nn.ModuleList()  
          
        # Bottom-up pathway (from lower level to higher level)  
        self.bu_convs = nn.ModuleList()  
        self.bu_weights = nn.ModuleList()  
          
        # Top-down pathway  
        for i in range(2):  # For 3 levels, we need 2 top-down connections  
            self.td_weights.append(WeightedFeatureFusion(2))  # Fuse 2 inputs  
            self.td_convs.append(  
                RepNCSPELAN4(  
                    hidden_dim,  
                    hidden_dim,  
                    hidden_dim,  
                    round(expansion * hidden_dim // 2),  
                    round(3 * depth_mult),  
                )  
            )  
          
        # Bottom-up pathway  
        for i in range(2):  # For 3 levels, we need 2 bottom-up connections  
            self.bu_weights.append(WeightedFeatureFusion(3))  # Fuse 3 inputs (original, top-down result, and lower level)  
            self.bu_convs.append(  
                RepNCSPELAN4(  
                    hidden_dim,  
                    hidden_dim,  
                    hidden_dim,  
                    round(expansion * hidden_dim // 2),  
                    round(3 * depth_mult),  
                )  
            )  
              
        # Downsample convs for bottom-up pathway  
        self.downsample_convs = nn.ModuleList()  
        for _ in range(2):  # For 3 levels, we need 2 downsampling operations  
            self.downsample_convs.append(  
                SCDown(hidden_dim, hidden_dim, 3, 2)  
            )  
      
    def forward(self, features):  
        # features should be a list from lowest resolution (highest level) to highest resolution (lowest level)  
        # For example: [P5, P4, P3] where P5 is 1/32, P4 is 1/16, P3 is 1/8  
        
        # Make a copy of the original features for the bottom-up path  
        original_features = features.copy()  
        
        # Top-down pathway (from P5 to P3)  
        td_features = [features[0]]  # Start with P5  
        
        for i in range(len(features) - 1):  
            # Get target shape from the next feature level  
            target_shape = features[i+1].shape[2:]  
            
            # Upsample higher level feature to match the exact spatial dimensions of the next level  
            upsample_feat = F.interpolate(td_features[-1], size=target_shape, mode="nearest")  
            
            # Weighted fusion of upsampled feature and original feature  
            fused = self.td_weights[i](upsample_feat, features[i+1])  
            
            # Apply convolution  
            td_feature = self.td_convs[i](fused)  
            td_features.append(td_feature)  
        
        # Bottom-up pathway (from P3 to P5)  
        bu_features = [td_features[-1]]  # Start with P3 from top-down path  
        
        for i in range(len(features) - 1):  
            # Downsample lower level feature  
            downsample_feat = self.downsample_convs[i](bu_features[-1])  
            
            # Get corresponding top-down feature and original feature  
            idx = len(features) - 2 - i  # Index for accessing the correct top-down feature  
            
            # Weighted fusion of downsampled feature, top-down feature, and original feature  
            fused = self.bu_weights[i](downsample_feat, td_features[idx], original_features[idx])  
            
            # Apply convolution  
            bu_feature = self.bu_convs[i](fused)  
            bu_features.append(bu_feature)  
        
        # Return features in the same order as input: [P5, P4, P3]  
        return list(reversed(bu_features))
  
  
@register()  
class HybridBiFPNEncoder(nn.Module):  
    """  
    Hybrid encoder that combines transformer encoding with BiFPN  
    """  
    __share__ = [  
        "eval_spatial_size",  
    ]  
  
    def __init__(  
        self,  
        in_channels=[512, 1024, 2048],  
        feat_strides=[8, 16, 32],  
        hidden_dim=256,  
        nhead=8,  
        dim_feedforward=1024,  
        dropout=0.0,  
        enc_act="gelu",  
        use_encoder_idx=[2],  
        num_encoder_layers=1,  
        pe_temperature=10000,  
        expansion=1.0,  
        depth_mult=1.0,  
        act="silu",  
        eval_spatial_size=None,  
        num_bifpn_blocks=3,  # Number of BiFPN blocks to stack  
    ):  
        super().__init__()  
        self.in_channels = in_channels  
        self.feat_strides = feat_strides  
        self.hidden_dim = hidden_dim  
        self.use_encoder_idx = use_encoder_idx  
        self.num_encoder_layers = num_encoder_layers  
        self.pe_temperature = pe_temperature  
        self.eval_spatial_size = eval_spatial_size  
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]  
        self.out_strides = feat_strides  
        self.num_bifpn_blocks = num_bifpn_blocks  
  
        # Channel projection  
        self.input_proj = nn.ModuleList()  
        for in_channel in in_channels:  
            proj = nn.Sequential(  
                OrderedDict(  
                    [  
                        ("conv", nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),  
                        ("norm", nn.BatchNorm2d(hidden_dim)),  
                    ]  
                )  
            )  
            self.input_proj.append(proj)  
  
        # Encoder transformer  
        if num_encoder_layers > 0:  
            encoder_layer = TransformerEncoderLayer(  
                hidden_dim,  
                nhead=nhead,  
                dim_feedforward=dim_feedforward,  
                dropout=dropout,  
                activation=enc_act,  
            )  
  
            self.encoder = nn.ModuleList(  
                [  
                    TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers)  
                    for _ in range(len(use_encoder_idx))  
                ]  
            )  
          
        # BiFPN blocks  
        self.bifpn_blocks = nn.ModuleList()  
        for _ in range(num_bifpn_blocks):  
            self.bifpn_blocks.append(  
                BiFPNBlock(hidden_dim, expansion, depth_mult)  
            )  
          
        self._reset_parameters()  
      
    def _reset_parameters(self):  
        # Initialize positional embeddings for evaluation  
        if self.eval_spatial_size:  
            for idx in self.use_encoder_idx:  
                stride = self.feat_strides[idx]  
                pos_embed = self.build_2d_sincos_position_embedding(  
                    self.eval_spatial_size[1] // stride,  
                    self.eval_spatial_size[0] // stride,  
                    self.hidden_dim,  
                    self.pe_temperature,  
                )  
                setattr(self, f"pos_embed{idx}", pos_embed)  
          
        # Initialize other parameters  
        for p in self.parameters():  
            if p.dim() > 1:  
                nn.init.xavier_uniform_(p)  
      
    def build_2d_sincos_position_embedding(self, w, h, embed_dim=256, temperature=10000):  
        grid_w = torch.arange(w, dtype=torch.float32)  
        grid_h = torch.arange(h, dtype=torch.float32)  
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")  
        assert embed_dim % 4 == 0, "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"  
        pos_dim = embed_dim // 4  
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim  
        omega = 1.0 / (temperature**omega)  
          
        # Fix: Use direct tensor arguments instead of a list  
        out_w = torch.einsum("m,d->md", grid_w.flatten(), omega)  
        out_h = torch.einsum("m,d->md", grid_h.flatten(), omega)  
          
        pos_emb = torch.cat(  
            [torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1  
        )[None, :, :]  
  
        return pos_emb  
      
    def forward(self, feats):  
        assert len(feats) == len(self.in_channels)  
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]  
      
        # Encoder  
        if self.num_encoder_layers > 0:  
            for i, enc_ind in enumerate(self.use_encoder_idx):  
                h, w = proj_feats[enc_ind].shape[2:]  
                # Flatten [B, C, H, W] to [B, HxW, C]  
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)  
                  
                # Fix: Ensure pos_embed is always assigned a value  
                if self.training or self.eval_spatial_size is None:  
                    pos_embed = self.build_2d_sincos_position_embedding(  
                        w, h, self.hidden_dim, self.pe_temperature  
                    ).to(src_flatten.device)  
                else:  
                    pos_embed_attr = getattr(self, f"pos_embed{enc_ind}", None)  
                    if pos_embed_attr is not None:  
                        pos_embed = pos_embed_attr.to(src_flatten.device)  
                    else:  
                        # Fallback if attribute doesn't exist  
                        pos_embed = self.build_2d_sincos_position_embedding(  
                            w, h, self.hidden_dim, self.pe_temperature  
                        ).to(src_flatten.device)  
      
                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)  
                proj_feats[enc_ind] = (  
                    memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()  
                )  
      
        # Apply BiFPN blocks  
        bifpn_feats = proj_feats  
        for bifpn_block in self.bifpn_blocks:  
            bifpn_feats = bifpn_block(bifpn_feats)  
          
        return bifpn_feats