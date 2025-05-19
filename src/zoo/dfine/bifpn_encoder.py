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
  
__all__ = ["BiFPNEncoder"]  
  
  
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
  
  
class SpatialAttention(nn.Module):  
    def __init__(self, in_channels):  
        super().__init__()  
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)  
          
    def forward(self, x):  
        # Create channel-wise attention  
        avg_pool = torch.mean(x, dim=1, keepdim=True)  
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  
        attention = torch.cat([avg_pool, max_pool], dim=1)  
        attention = self.conv(attention)  
        attention = torch.sigmoid(attention)  
        return x * attention  
  
  
class WeightedFeatureFusion(nn.Module):  
    def __init__(self, in_channels, out_channels, num_inputs=2, use_attention=False):  
        super().__init__()  
        self.in_channels = in_channels  
        self.out_channels = out_channels  
        self.num_inputs = num_inputs  
        self.use_attention = use_attention  
          
        # Learnable weights for feature fusion  
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32), requires_grad=True)  
          
        # Optional spatial attention  
        if use_attention:  
            self.attention = SpatialAttention(out_channels)  
          
        # 1x1 conv to adjust channel dimensions if needed  
        if in_channels != out_channels:  
            self.conv = ConvNormLayer_fuse(in_channels, out_channels, 1, 1, act="silu")  
        else:  
            self.conv = nn.Identity()  
      
    def forward(self, inputs):  
        # Apply softmax to weights for normalization  
        weights = F.softmax(self.weights, dim=0)  
          
        # Weighted sum of inputs  
        x = torch.zeros_like(inputs[0])  
        for i, inp in enumerate(inputs):  
            x = x + weights[i] * inp  
              
        # Apply channel adjustment if needed  
        x = self.conv(x)  
          
        # Apply spatial attention if enabled  
        if self.use_attention:  
            x = self.attention(x)  
              
        return x  
  
  
class BiFPNBlock(nn.Module):  
    def __init__(  
        self,   
        channels,   
        num_levels=3,   
        weighted_fusion=True,  
        use_attention=False,  
        act="silu"  
    ):  
        super().__init__()  
        self.num_levels = num_levels  
        self.weighted_fusion = weighted_fusion  
          
        # Top-down pathway (from higher level to lower level)  
        self.top_down_convs = nn.ModuleList()  
        self.top_down_fusions = nn.ModuleList()  
          
        # Bottom-up pathway (from lower level to higher level)  
        self.bottom_up_convs = nn.ModuleList()  
        self.bottom_up_fusions = nn.ModuleList()  
          
        # Top-down pathway  
        for i in range(num_levels - 1, 0, -1):  
            # 1x1 conv for lateral connection  
            self.top_down_convs.append(  
                ConvNormLayer_fuse(channels, channels, 1, 1, act=act)  
            )  
              
            # Feature fusion  
            if weighted_fusion:  
                self.top_down_fusions.append(  
                    WeightedFeatureFusion(channels, channels, num_inputs=2, use_attention=use_attention)  
                )  
            else:  
                self.top_down_fusions.append(  
                    ConvNormLayer_fuse(channels * 2, channels, 1, 1, act=act)  
                )  
          
        # Bottom-up pathway  
        for i in range(0, num_levels - 1):  
            # Downsample conv  
            self.bottom_up_convs.append(  
                ConvNormLayer_fuse(channels, channels, 3, 2, act=act)  
            )  
              
            # Feature fusion  
            if weighted_fusion:  
                self.bottom_up_fusions.append(  
                    WeightedFeatureFusion(channels, channels, num_inputs=3 if i > 0 else 2, use_attention=use_attention)  
                )  
            else:  
                self.bottom_up_fusions.append(  
                    ConvNormLayer_fuse(channels * (3 if i > 0 else 2), channels, 1, 1, act=act)  
                )  
      
    def forward(self, features):  
        # features is a list of feature maps from different levels  
        # features[0] is the lowest level (highest resolution)  
        # features[-1] is the highest level (lowest resolution)  
          
        # Store original features for skip connections  
        original_features = features.copy()  
          
        # Top-down pathway (from higher level to lower level)  
        for i in range(len(features) - 1, 0, -1):  
            td_idx = len(features) - 1 - i  
              
            # Upsample higher level feature  
            higher_feat = F.interpolate(  
                features[i], scale_factor=2.0, mode="nearest"  
            )  
              
            # Process lower level feature  
            lower_feat = self.top_down_convs[td_idx](features[i-1])  
              
            # Fusion  
            if self.weighted_fusion:  
                features[i-1] = self.top_down_fusions[td_idx]([lower_feat, higher_feat])  
            else:  
                features[i-1] = self.top_down_fusions[td_idx](  
                    torch.cat([lower_feat, higher_feat], dim=1)  
                )  
          
        # Bottom-up pathway (from lower level to higher level)  
        for i in range(len(features) - 1):  
            # Downsample lower level feature  
            lower_feat = self.bottom_up_convs[i](features[i])  
              
            # Get original feature at this level for skip connection  
            orig_feat = original_features[i+1]  
              
            # For levels after the first, also include the processed feature from the previous iteration  
            if i > 0:  
                if self.weighted_fusion:  
                    features[i+1] = self.bottom_up_fusions[i]([orig_feat, lower_feat, features[i+1]])  
                else:  
                    features[i+1] = self.bottom_up_fusions[i](  
                        torch.cat([orig_feat, lower_feat, features[i+1]], dim=1)  
                    )  
            else:  
                if self.weighted_fusion:  
                    features[i+1] = self.bottom_up_fusions[i]([orig_feat, lower_feat])  
                else:  
                    features[i+1] = self.bottom_up_fusions[i](  
                        torch.cat([orig_feat, lower_feat], dim=1)  
                    )  
          
        return features  
  
  
@register()  
class BiFPNEncoder(nn.Module):  
    __share__ = [  
        "eval_spatial_size",  
    ]  
  
    def __init__(  
        self,  
        in_channels=[512, 1024, 2048],  
        feat_strides=[8, 16, 32],  
        hidden_dim=256,  
        num_bifpn_layers=3,  
        weighted_fusion=True,  
        use_attention=False,  
        attention_type=None,  
        dropout=0.0,  
        act="silu",  
        eval_spatial_size=None,  
    ):  
        super().__init__()  
        self.in_channels = in_channels  
        self.feat_strides = feat_strides  
        self.hidden_dim = hidden_dim  
        self.num_bifpn_layers = num_bifpn_layers  
        self.weighted_fusion = weighted_fusion  
        self.use_attention = use_attention or (attention_type is not None)  
        self.eval_spatial_size = eval_spatial_size  
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]  
        self.out_strides = feat_strides  
  
        # Channel projection to get uniform channel dimensions  
        self.input_proj = nn.ModuleList()  
        for in_channel in in_channels:  
            proj = nn.Sequential(  
                OrderedDict(  
                    [  
                        ("conv", nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),  
                        ("norm", nn.BatchNorm2d(hidden_dim)),  
                        ("act", get_activation(act)),  
                    ]  
                )  
            )  
            self.input_proj.append(proj)  
  
        # BiFPN blocks  
        self.bifpn_blocks = nn.ModuleList()  
        for _ in range(num_bifpn_layers):  
            self.bifpn_blocks.append(  
                BiFPNBlock(  
                    channels=hidden_dim,  
                    num_levels=len(in_channels),  
                    weighted_fusion=weighted_fusion,  
                    use_attention=self.use_attention,  
                    act=act  
                )  
            )  
  
    def forward(self, feats):  
        assert len(feats) == len(self.in_channels)  
          
        # Project input features to the same channel dimension  
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]  
          
        # Apply BiFPN blocks  
        for bifpn_block in self.bifpn_blocks:  
            proj_feats = bifpn_block(proj_feats)  
          
        return proj_feats