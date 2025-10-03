from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers

from .ops import ConvBNReLU, resize_to

class MAD(nn.Module):
    def __init__(self, in_c, num_groups=4, hidden_dim=None):
        super().__init__()
        self.num_groups = num_groups
        hidden_dim = hidden_dim or in_c // 2
        expand_dim = hidden_dim * num_groups
        self.expand_conv = ConvBNReLU(in_c, expand_dim, 1)
        self.interact = nn.ModuleDict()
        self.interact["0"] = ConvBNReLU(hidden_dim, 2 * hidden_dim, 3, 1, 1)
        for group_id in range(1, num_groups - 1):
            self.interact[str(group_id)] = ConvBNReLU(2 * hidden_dim, 2 * hidden_dim, 3, 1, 1)
        self.interact[str(num_groups - 1)] = ConvBNReLU(2 * hidden_dim, 1 * hidden_dim, 3, 1, 1)
        self.fuse = nn.Sequential(nn.Conv2d(num_groups * hidden_dim, in_c, 3, 1, 1), nn.BatchNorm2d(in_c))
        self.final_relu = nn.ReLU(True)

    def forward(self, x):
        xs = self.expand_conv(x).chunk(self.num_groups, dim=1)
        outs = []
        branch_out = self.interact["0"](xs[0])
        outs.append(branch_out.chunk(2, dim=1))

        for group_id in range(1, self.num_groups - 1):
            branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
            outs.append(branch_out.chunk(2, dim=1))

        group_id = self.num_groups - 1
        branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
        outs.append(branch_out.chunk(1, dim=1))
        out = torch.cat([o[0] for o in outs], dim=1)
        out = self.fuse(out)
        return self.final_relu(out + x)


class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super(ResidualConvUnit, self).__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x

class FeatureFusionBlock(nn.Module):
    def __init__(self, features):
        super(FeatureFusionBlock, self).__init__()
        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)
    def forward(self, *xs):
        output = xs[0]
        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])
        output = self.resConfUnit2(output)
        output = F.interpolate(output, scale_factor=2, mode="bilinear", align_corners=True)
        return output

def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.Sigmoid, nn.Softmax, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool1d, nn.Sigmoid, nn.Identity)):
            pass
        else:
            m.initialize()

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

    def initialize(self):
        weight_init(self)

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

    def initialize(self):
        weight_init(self)

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, mode):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv_0 = nn.Conv2d(dim * 3, dim* 3, kernel_size=1, bias=bias)
        self.qkv_1 = nn.Conv2d(dim* 3, dim* 3, kernel_size=1, bias=bias)
        self.qkv_2 = nn.Conv2d(dim* 3, dim* 3, kernel_size=1, bias=bias)

        self.qkv1conv_3 = nn.Conv2d(dim* 3, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv2conv_3 = nn.Conv2d(dim* 3, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv3conv_3 = nn.Conv2d(dim* 3, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.conv_l_pre = ConvBNReLU(dim,dim, 3, 1, 1)
        self.conv_s_pre = ConvBNReLU(dim, dim, 3, 1, 1)
        self.conv_l = ConvBNReLU(dim, dim, 3, 1, 1)  # intra-branch
        self.conv_m = ConvBNReLU(dim, dim, 3, 1, 1)  # intra-branch
        self.conv_s = ConvBNReLU(dim, dim, 3, 1, 1)  # intra-branch

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, l,m,s):
        tgt_size = m.shape[2:]

        l = self.conv_l_pre(l)
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        s = self.conv_s_pre(s)
        s = resize_to(s, tgt_hw=m.shape[2:])
        l = self.conv_l(l)
        m = self.conv_m(m)
        s = self.conv_s(s)
        lms = torch.cat([l, m, s], dim=1)  # BT,3C,H,W

        b, c, hl, wl = lms.shape
        lmsq_3 = self.qkv1conv_3(self.qkv_0(lms))
        lmsk_3 = self.qkv2conv_3(self.qkv_1(lms))
        lmsv_3 = self.qkv3conv_3(self.qkv_2(lms))

        q_3 = rearrange(lmsq_3, 'b (head c) hl wl -> b head c (hl wl)', head=self.num_heads)
        k_3 = rearrange(lmsk_3, 'b (head c) hl wl -> b head c (hl wl)', head=self.num_heads)
        v_3 = rearrange(lmsv_3, 'b (head c) hl wl -> b head c (hl wl)', head=self.num_heads)

        q_3 = torch.nn.functional.normalize(q_3, dim=-1)
        k_3 = torch.nn.functional.normalize(k_3, dim=-1)
        attn_3 = (q_3 @ k_3.transpose(-2, -1)) * self.temperature
        attn_3 = attn_3.softmax(dim=-1)
        out_3 = (attn_3 @ v_3)
        out_3 = rearrange(out_3, 'b head c (hl wl) -> b (head c) hl wl', head=self.num_heads, hl=hl, wl=wl)
        out = self.project_out(out_3)
        return out

    def initialize(self):
        weight_init(self)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv_3 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.dwconv_5 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=5, stride=1, padding=2,
                                groups=hidden_features * 2, bias=bias)
        self.dwconv_7 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=7, stride=1, padding=3,
                                  groups=hidden_features * 2, bias=bias)

        self.dilate3 = nn.Sequential(
            nn.Conv2d(hidden_features * 2, hidden_features * 2, 3, padding=3, dilation=3), nn.BatchNorm2d(hidden_features * 2),
        )
        self.dilate5 = nn.Sequential(
            nn.Conv2d(hidden_features * 2, hidden_features * 2, 3, padding=5, dilation=5), nn.BatchNorm2d(hidden_features * 2),
        )
        self.dilate7 = nn.Sequential(
            nn.Conv2d(hidden_features * 2, hidden_features * 2, 3, padding=7, dilation=7), nn.BatchNorm2d(hidden_features * 2),
        )
        self.reduce = nn.Sequential(
            nn.Conv2d(512, hidden_features, 3, 1, 1),nn.BatchNorm2d(hidden_features),nn.ReLU(True)
        )
        self.project_out = nn.Conv2d(hidden_features*3, dim, kernel_size=1, bias=bias)

    def forward(self, lms):
        lms_comm = self.project_in(lms)
        d1_3, d2_3 = self.dwconv_3(lms_comm).chunk(2, dim=1)
        d_3 = F.gelu(d1_3) * d2_3
        d1_5, d2_5 = self.dwconv_5(lms_comm).chunk(2, dim=1)
        d_5 = F.gelu(d1_5) * d2_5
        d1_7, d2_7 = self.dwconv_7(lms_comm).chunk(2, dim=1)
        d_7 = F.gelu(d1_7) * d2_7


        x = self.project_out(torch.cat((d_3,d_5,d_7),1))
        return x

    def initialize(self):
        weight_init(self)

class MGFF(nn.Module): # Multi-scale transformer block
    def __init__(self, mode='dilation', dim=128, num_heads=8, ffn_expansion_factor=4, bias=False,
                 LayerNorm_type='WithBias'):
        super(MGFF, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias, mode)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, l,m,s):
        l = self.norm1(l)
        m = self.norm1(m)
        s = self.norm1(s)
        lms = m + self.attn(l,m,s)

        x = lms + self.ffn(self.norm2(lms))
        return x

class SimpleASPP(nn.Module):
    def __init__(self, in_dim, out_dim, dilation=3):
        super().__init__()
        self.conv1x1_1 = ConvBNReLU(in_dim, 2 * out_dim, 1)
        self.conv1x1_2 = ConvBNReLU(out_dim, out_dim, 1)
        self.conv3x3_1 = ConvBNReLU(out_dim, out_dim, 3, dilation=dilation, padding=dilation)
        self.conv3x3_2 = ConvBNReLU(out_dim, out_dim, 3, dilation=dilation, padding=dilation)
        self.conv3x3_3 = ConvBNReLU(out_dim, out_dim, 3, dilation=dilation, padding=dilation)
        self.fuse = nn.Sequential(ConvBNReLU(5 * out_dim, out_dim, 1), ConvBNReLU(out_dim, out_dim, 3, 1, 1))

    def forward(self, x):
        y = self.conv1x1_1(x)
        y1, y5 = y.chunk(2, dim=1)

        # dilation branch
        y2 = self.conv3x3_1(y1)
        y3 = self.conv3x3_2(y2)
        y4 = self.conv3x3_3(y3)

        # global branch
        y0 = torch.mean(y5, dim=(2, 3), keepdim=True)
        y0 = self.conv1x1_2(y0)
        y0 = resize_to(y0, tgt_hw=x.shape[-2:])
        return self.fuse(torch.cat([y0, y1, y2, y3, y4], dim=1))
