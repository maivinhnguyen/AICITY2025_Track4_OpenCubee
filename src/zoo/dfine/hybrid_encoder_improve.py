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

__all__ = ["HybridEncoderImprove"]


class ConvNormLayer_fuse(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size - 1) // 2 if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in, ch_out, kernel_size, stride, groups=g, padding=padding, bias=bias
        )
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)
        self.ch_in, self.ch_out, self.kernel_size, self.stride, self.g, self.padding, self.bias = (
            ch_in,
            ch_out,
            kernel_size,
            stride,
            g,
            padding,
            bias,
        )

    def forward(self, x):
        if hasattr(self, "conv_bn_fused"):
            y = self.conv_bn_fused(x)
        else:
            y = self.norm(self.conv(x))
        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, "conv_bn_fused"):
            self.conv_bn_fused = nn.Conv2d(
                self.ch_in,
                self.ch_out,
                self.kernel_size,
                self.stride,
                groups=self.g,
                padding=self.padding,
                bias=True,
            )

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv_bn_fused.weight.data = kernel
        self.conv_bn_fused.bias.data = bias
        self.__delattr__("conv")
        self.__delattr__("norm")

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor()

        return kernel3x3, bias3x3

    def _fuse_bn_tensor(self):
        kernel = self.conv.weight
        running_mean = self.norm.running_mean
        running_var = self.norm.running_var
        gamma = self.norm.weight
        beta = self.norm.bias
        eps = self.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size - 1) // 2 if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in, ch_out, kernel_size, stride, groups=g, padding=padding, bias=bias
        )
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class SCDown(nn.Module):
    def __init__(self, c1, c2, k, s):
        super().__init__()
        self.cv1 = ConvNormLayer_fuse(c1, c2, 1, 1)
        self.cv2 = ConvNormLayer_fuse(c2, c2, k, s, c2)

    def forward(self, x):
        return self.cv2(self.cv1(x))


class VGGBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act="relu"):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else act

    def forward(self, x):
        if hasattr(self, "conv"):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, "conv"):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        self.__delattr__("conv1")
        self.__delattr__("conv2")

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class ELAN(nn.Module):
    # csp-elan
    def __init__(self, c1, c2, c3, c4, n=2, bias=False, act="silu", bottletype=VGGBlock):
        super().__init__()
        self.c = c3
        self.cv1 = ConvNormLayer_fuse(c1, c3, 1, 1, bias=bias, act=act)
        self.cv2 = nn.Sequential(
            bottletype(c3 // 2, c4, act=get_activation(act)),
            ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act),
        )
        self.cv3 = nn.Sequential(
            bottletype(c4, c4, act=get_activation(act)),
            ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act),
        )
        self.cv4 = ConvNormLayer_fuse(c3 + (2 * c4), c2, 1, 1, bias=bias, act=act)

    def forward(self, x):
        # y = [self.cv1(x)]
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class RepNCSPELAN4(nn.Module):
    # csp-elan
    def __init__(self, c1, c2, c3, c4, n=3, bias=False, act="silu"):
        super().__init__()
        self.c = c3 // 2
        self.cv1 = ConvNormLayer_fuse(c1, c3, 1, 1, bias=bias, act=act)
        self.cv2 = nn.Sequential(
            CSPLayer(c3 // 2, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock),
            ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act),
        )
        self.cv3 = nn.Sequential(
            CSPLayer(c4, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock),
            ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act),
        )
        self.cv4 = ConvNormLayer_fuse(c3 + (2 * c4), c2, 1, 1, bias=bias, act=act)

    def forward_chunk(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class CSPLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_blocks=3,
        expansion=1.0,
        bias=False,
        act="silu",
        bottletype=VGGBlock,
    ):
        super(CSPLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(
            *[
                bottletype(hidden_channels, hidden_channels, act=get_activation(act))
                for _ in range(num_blocks)
            ]
        )
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer_fuse(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)

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
            self.bu_weights.append(WeightedFeatureFusion(3))  # Fuse 3 inputs  
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
              
            # Ensure all features have the same spatial dimensions  
            target_shape = original_features[idx].shape[2:]  
            if downsample_feat.shape[2:] != target_shape:  
                downsample_feat = F.interpolate(downsample_feat, size=target_shape, mode="nearest")  
              
            if td_features[idx].shape[2:] != target_shape:  
                td_feat_resized = F.interpolate(td_features[idx], size=target_shape, mode="nearest")  
            else:  
                td_feat_resized = td_features[idx]  
              
            # Weighted fusion of downsampled feature, top-down feature, and original feature  
            fused = self.bu_weights[i](downsample_feat, td_feat_resized, original_features[idx])  
              
            # Apply convolution  
            bu_feature = self.bu_convs[i](fused)  
            bu_features.append(bu_feature)  
          
        # Return features in the same order as input: [P5, P4, P3]  
        return list(reversed(bu_features))  


# transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


@register()  
class HybridEncoderImprove(nn.Module):  
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
        num_bifpn_blocks=3,  # New parameter for BiFPN  
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
        self.num_bifpn_blocks = num_bifpn_blocks  # Store the number of BiFPN blocks  
  
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
          
        # BiFPN blocks instead of FPN and PAN  
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
          
        # Use direct tensor arguments instead of a list for calflops compatibility  
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
                
                # Ensure pos_embed is always assigned a value  
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