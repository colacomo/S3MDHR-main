import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import to_2tuple, trunc_normal_
from utils.lossfunc import HyperspectralSWTLoss, SAMLoss, BandWiseMSE
import math
from models.SMDHR_modules import AdapterLayer

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y



class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 3, stride=1, padding=3, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return x * attn


# ====================== 1. Spectral-Aware Degradation Routing ======================
class F_ext(nn.Module):
    def __init__(self, in_nc=3, nf=64):
        super(F_ext, self).__init__()
        stride = 2
        pad = 0
        self.pad = nn.ZeroPad2d(1)
        self.conv1 = nn.Conv2d(in_nc, nf, 2, stride, pad, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 2, stride, pad, bias=True)
        self.conv3 = nn.Conv2d(nf, nf, 2, stride, pad, bias=True)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x):
        conv1_out = self.act(self.conv1(self.pad(x)))
        conv2_out = self.act(self.conv2(self.pad(conv1_out)))
        conv3_out = self.act(self.conv3(self.pad(conv2_out)))
        out = torch.mean(conv3_out, dim=[2, 3], keepdim=False)
        return out


# ====================== 2. Dynamic TopK Prediction Network ======================
class DynamicTopKPredictor(nn.Module):
    """Top-K predictor optimized by meta-learning."""

    def __init__(self, in_channels):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(in_channels, in_channels // 64),
            nn.ReLU(),
            nn.Linear(in_channels // 64, in_channels // 64),
            nn.ReLU(),
            nn.Linear(in_channels // 64, 2),
            nn.Tanh()
        )
        # Variant 6: adjust the initial ratio from [0.5, 0.9] to [0.5, 0.5]
        self.register_buffer('start', torch.tensor([0.5, 0.5], dtype=torch.float32))

    def forward(self, x):
        k_ratios = self.predictor(x)
        k_ratios = self.start.unsqueeze(0) + 0.4 * k_ratios
        k_ratios = k_ratios.clamp_(0, 1)
        return k_ratios


class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(SKFusion, self).__init__()
        self.height = height
        d = max(int(dim / reduction), 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape
        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)
        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))
        out = torch.sum(in_feats * attn, dim=1)
        return out


class PromptAdapter(nn.Module):
    def __init__(self, in_dim, act=nn.ReLU(), bias=False):
        super(PromptAdapter, self).__init__()
        self.linear_dw = nn.Linear(in_dim, in_dim // 8, bias=bias)
        self.act = act
        self.linear_up = nn.Linear(in_dim // 8, in_dim, bias=bias)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x):
        res = x
        x = self.linear_dw(x)
        x = self.act(x)
        x = self.linear_up(x)
        x = self.act(self.norm(x) + res)
        return x


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=True):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.MLP = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.LeakyReLU(),
            nn.Linear(in_channels * 2, out_channels * (1 + self.use_affine_level)),
        )
        self.adapter = PromptAdapter(512, act=nn.LeakyReLU(), bias=True)

    def forward(self, x, text_embed):
        text_embed = self.adapter(text_embed)
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.MLP(text_embed).view(batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        return x


class Fusion_Embed(nn.Module):
    def __init__(self, embed_dim, bias=False):
        super(Fusion_Embed, self).__init__()
        self.fusion_proj = nn.Conv2d(
            embed_dim * 2, embed_dim, kernel_size=1, stride=1, bias=bias
        )

    def forward(self, x_A, x_B):
        x = torch.concat([x_A, x_B], dim=1)
        x = self.fusion_proj(x)
        return x


class PatchEmbedmain(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        if kernel_size is None:
            kernel_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchUnEmbedmain(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        if kernel_size is None:
            kernel_size = 1
        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class RLN(nn.Module):
    def __init__(self, dim, eps=1e-5, detach_grad=False):
        super(RLN, self).__init__()
        self.eps = eps
        self.detach_grad = detach_grad
        self.weight = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.bias = nn.Parameter(torch.zeros((1, dim, 1, 1)))
        self.meta1 = nn.Conv2d(1, dim, 1)
        self.meta2 = nn.Conv2d(1, dim, 1)
        trunc_normal_(self.meta1.weight, std=.02)
        nn.init.constant_(self.meta1.bias, 1)
        trunc_normal_(self.meta2.weight, std=.02)
        nn.init.constant_(self.meta2.bias, 0)

    def forward(self, input):
        mean = torch.mean(input, dim=(1, 2, 3), keepdim=True)
        std = torch.sqrt((input - mean).pow(2).mean(dim=(1, 2, 3), keepdim=True) + self.eps)
        normalized_input = (input - mean) / std

        if self.detach_grad:
            rescale, rebias = self.meta1(std.detach()), self.meta2(mean.detach())
        else:
            rescale, rebias = self.meta1(std), self.meta2(mean)

        out = normalized_input * self.weight + self.bias
        return out, rescale, rebias


class Mlp(nn.Module):
    def __init__(self, network_depth, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.network_depth = network_depth
        self.mlp = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1),
            nn.ReLU(False),
            nn.Conv2d(hidden_features, out_features, 1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            gain = (8 * self.network_depth) ** (-1 / 4)
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size ** 2, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def get_relative_positions(window_size):
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
    coords_flatten = torch.flatten(coords, 1)
    relative_positions = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_positions = relative_positions.permute(1, 2, 0).contiguous()
    relative_positions_log = torch.sign(relative_positions) * torch.log(1. + relative_positions.abs())
    return relative_positions_log


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, top_k_ratio=0.5, dynamic_k=False):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.dynamic_k = dynamic_k

        relative_positions = get_relative_positions(self.window_size)
        self.register_buffer("relative_positions", relative_positions)
        self.meta = nn.Sequential(
            nn.Linear(2, 256, bias=True),
            nn.ReLU(False),
            nn.Linear(256, num_heads, bias=True)
        )
        self.softmax = nn.Softmax(dim=-1)
        self.top_k_ratio = top_k_ratio if not dynamic_k else None

    def forward(self, qkv, k_ratio=None):
        B_, N, _ = qkv.shape
        qkv = qkv.reshape(B_, N, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.meta(self.relative_positions)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        # Variant 6: use the passed-in dynamic ratio
        top_k_ratio = k_ratio if self.dynamic_k and k_ratio is not None else self.top_k_ratio
        if top_k_ratio is None:
            top_k_ratio = 0.5

        if isinstance(top_k_ratio, torch.Tensor):
            top_k = torch.clamp((N * top_k_ratio).long(), min=1)
            top_k = top_k.view(-1, 1, 1, 1)
            Batch, _, _, _ = top_k.shape
            mask = torch.zeros_like(attn)
            for b in range(top_k.shape[0]):
                k = top_k[b].item()
                dim = attn.shape[0] // top_k.shape[0]
                if k > 0:
                    top_k_values, top_k_indices = torch.topk(attn[b:b + dim], k=k, dim=-1, largest=True)
                    mask[b * dim:(b + 1) * dim].scatter_(-1, top_k_indices, 1.)
            attn = torch.where(mask > 0, attn, torch.full_like(attn, float('-inf')))
        else:
            top_k = max(int(N * top_k_ratio), 1) if top_k_ratio != 0 else N
            if top_k < N:
                top_k_values, top_k_indices = torch.topk(attn, k=top_k, dim=-1, largest=True)
                mask = torch.zeros_like(attn).scatter_(-1, top_k_indices, 1.)
                attn = torch.where(mask > 0, attn, torch.full_like(attn, float('-inf')))

        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, self.dim)
        return x


class GSAttention_TopK(nn.Module):
    def __init__(self, dim, num_heads=1, top_k_ratio=0.5, dynamic_k=False):
        super().__init__()
        self.num_heads = num_heads
        self.top_k_ratio = top_k_ratio if not dynamic_k else None
        self.dynamic_k = dynamic_k
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x, k_spe=None):
        b, c, h, w = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        # Variant 6: use the passed-in dynamic ratio
        top_k_ratio = k_spe if self.dynamic_k and k_spe is not None else self.top_k_ratio
        if top_k_ratio is None:
            top_k_ratio = 0.5

        c_per_head = c // self.num_heads

        # Variant 6: fix the dynamic threshold to prevent out-of-bounds
        if isinstance(top_k_ratio, torch.Tensor):
            top_k_tensor = torch.clamp((c_per_head * top_k_ratio).long(), min=1)
            top_k_tensor = top_k_tensor.view(-1, 1, 1, 1)
            mask = torch.zeros_like(attn)
            for bi in range(b):
                k_val = top_k_tensor[bi].item()
                if k_val < c_per_head:
                    _, idx = torch.topk(attn[bi], k=k_val, dim=-1, largest=True)
                    mask[bi].scatter_(-1, idx, 1.)
                else:
                    mask[bi] = 1.0  # keep all
            attn = torch.where(mask.bool(), attn, torch.full_like(attn, float('-inf')))
        else:
            top_k = max(int(c_per_head * top_k_ratio), 1)
            if top_k < c_per_head:
                _, idx = torch.topk(attn, k=top_k, dim=-1, largest=True)
                mask = torch.zeros_like(attn).scatter_(-1, idx, 1.)
                attn = torch.where(mask.bool(), attn, torch.full_like(attn, float('-inf')))

        attn = attn.softmax(dim=-1)
        weight = (attn @ v)
        weight = weight.mean(dim=-1, keepdim=True)
        weight = rearrange(weight, 'b head c 1 -> b (head c) 1 1')
        return weight


class Attention(nn.Module):
    def __init__(self, network_depth, dim, num_heads, window_size, shift_size, use_attn=False, conv_type=None,
                 top_k_spa=0.5, top_k_spe=0.5, dynamic_k=False):
        super().__init__()
        self.dim = dim
        self.head_dim = int(dim // num_heads)
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.network_depth = network_depth
        self.use_attn = use_attn
        self.conv_type = conv_type
        self.dynamic_k = dynamic_k

        if self.conv_type == 'Conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.ReLU(False),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect')
            )

        if self.conv_type == 'DWConv':
            self.conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode='reflect')

        if self.conv_type == 'DWConv' or self.use_attn:
            self.V = nn.Conv2d(dim, dim, 1)
            self.proj = nn.Conv2d(dim, dim, 1)

        if self.use_attn:
            self.QK = nn.Conv2d(dim, dim * 2, 1)
            # Variant 6: explicitly pass the config computed at the top level during initialization
            self.attn = WindowAttention(dim, window_size, num_heads, top_k_ratio=top_k_spa, dynamic_k=dynamic_k)
            self.QKV_spec = nn.Conv2d(dim, dim * 3, 1)
            self.specattn = GSAttention_TopK(dim, num_heads, top_k_ratio=top_k_spe, dynamic_k=dynamic_k)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            w_shape = m.weight.shape
            if w_shape[0] == self.dim * 2:
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)
            else:
                gain = (8 * self.network_depth) ** (-1 / 4)
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def check_size(self, x, shift=False):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        if shift:
            x = F.pad(x, (self.shift_size, (self.window_size - self.shift_size + mod_pad_w) % self.window_size,
                          self.shift_size, (self.window_size - self.shift_size + mod_pad_h) % self.window_size),
                      mode='reflect')
        else:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, X, k_spa=None, k_spe=None):
        B, C, H, W = X.shape
        if self.conv_type == 'DWConv' or self.use_attn:
            V = self.V(X)
        if self.use_attn:
            QK = self.QK(X)
            QKV = torch.cat([QK, V], dim=1)
            shifted_QKV = self.check_size(QKV, self.shift_size > 0)
            Ht, Wt = shifted_QKV.shape[2:]
            shifted_QKV = shifted_QKV.permute(0, 2, 3, 1)
            qkv = window_partition(shifted_QKV, self.window_size)
            # Variant 6: forward the parameters downstream
            attn_windows = self.attn(qkv, k_spa)
            shifted_out = window_reverse(attn_windows, self.window_size, Ht, Wt)
            out = shifted_out[:, self.shift_size:(self.shift_size + H), self.shift_size:(self.shift_size + W), :]
            attn_out = out.permute(0, 3, 1, 2)
            if self.conv_type in ['Conv', 'DWConv']:
                conv_out = self.conv(V)
                attn_out = conv_out + attn_out
                # Variant 6: forward the parameters downstream
                spe_weight = self.specattn(attn_out, k_spe)
                attn_out = attn_out * spe_weight + attn_out
                out = self.proj(attn_out)
            else:
                out = self.proj(attn_out)
        else:
            if self.conv_type == 'Conv':
                out = self.conv(X)
            elif self.conv_type == 'DWConv':
                out = self.proj(self.conv(V))
        return out


class TopKTransformerBlock(nn.Module):
    def __init__(self, network_depth, dim, num_heads, mlp_ratio=4., norm_layer=nn.LayerNorm, mlp_norm=False,
                 window_size=8, shift_size=0, use_attn=True, conv_type=None, top_k_spa=0.5, top_k_spe=0.5,
                 dynamic_k=False):
        super().__init__()
        self.use_attn = use_attn
        self.mlp_norm = mlp_norm
        self.norm1 = norm_layer(dim) if use_attn else nn.Identity()
        self.attn = Attention(network_depth, dim, num_heads=num_heads, window_size=window_size, shift_size=shift_size,
                              use_attn=use_attn, conv_type=conv_type, top_k_spa=top_k_spa, top_k_spe=top_k_spe,
                              dynamic_k=dynamic_k)
        self.norm2 = norm_layer(dim) if use_attn and mlp_norm else nn.Identity()
        self.mlp = Mlp(network_depth, dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x, k_spa=None, k_spe=None):
        identity = x
        if self.use_attn:
            x, rescale, rebias = self.norm1(x)
        # pass-through
        x = self.attn(x, k_spa=k_spa, k_spe=k_spe)
        if self.use_attn:
            x = x * rescale + rebias
        x = identity + x
        identity = x
        if self.use_attn and self.mlp_norm:
            x, rescale, rebias = self.norm2(x)
        x = self.mlp(x)
        if self.use_attn and self.mlp_norm:
            x = x * rescale + rebias
        x = identity + x
        return x


class TSSTLayer(nn.Module):
    def __init__(self, network_depth, dim, depth, num_heads, mlp_ratio=4., norm_layer=nn.LayerNorm, window_size=8,
                 attn_ratio=0., attn_loc='last', conv_type=None, top_k_spa=0.5, top_k_spe=0.5, dynamic_k=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        attn_depth = attn_ratio * depth
        if attn_loc == 'last':
            use_attns = [i >= depth - attn_depth for i in range(depth)]
        elif attn_loc == 'first':
            use_attns = [i < attn_depth for i in range(depth)]
        elif attn_loc == 'middle':
            use_attns = [i >= (depth - attn_depth) // 2 and i < (depth + attn_depth) // 2 for i in range(depth)]
        self.blocks = nn.ModuleList([
            TopKTransformerBlock(network_depth=network_depth, dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                 norm_layer=norm_layer, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 use_attn=use_attns[i], conv_type=conv_type, top_k_spa=top_k_spa, top_k_spe=top_k_spe,
                                 dynamic_k=dynamic_k)
            for i in range(depth)])

    def forward(self, x, k_spa=None, k_spe=None):
        for blk in self.blocks:
            x = blk(x, k_spa=k_spa, k_spe=k_spe)
        return x


class FRMoELayer(nn.Module):
    def __init__(self, dim, depth, stage_depth=1, top_kexp=3, top_k_ratio=0.5, dynamic_k=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.blocks = nn.ModuleList([
            AdapterLayer(
                dim, top_k=top_kexp, num_experts=10, freq_dim=512, stage_depth=stage_depth,
                with_complexity=True, complexity_scale="max"
            )
            for i in range(depth)])

    def forward(self, x, emb, degra_map):
        for blk in self.blocks:
            x = blk(x, emb, degra_map)
        return x


class FeaFusAttenBlock(nn.Module):
    def __init__(self, channel):
        super(FeaFusAttenBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel * 2, channel, 1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(channel, channel, 3, padding=1, bias=True)
        self.ca1 = CALayer(channel)
        self.lka1 = LKA(channel)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.ca1(y)
        y = self.lka1(y)
        return y


class Cross_attention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=16, top_k_ratio=0.5, dynamic_k=False):
        super().__init__()
        self.n_head = n_head
        self.norm_A = nn.GroupNorm(norm_groups, in_channel)
        self.norm_B = nn.GroupNorm(norm_groups, in_channel)
        self.qkv_A = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out_A = nn.Conv2d(in_channel, in_channel, 1)
        self.qkv_B = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out_B = nn.Conv2d(in_channel, in_channel, 1)
        self.dynamic_k = dynamic_k
        self.top_k_ratio = top_k_ratio if not dynamic_k else None

    def forward(self, x_A, x_B, k_ratio=None):
        batch, channel, height, width = x_A.shape
        n_head = self.n_head
        head_dim = channel // n_head

        x_A = self.norm_A(x_A)
        query_A, key_A, value_A = self.qkv_A(x_A).view(batch, n_head, head_dim * 3, height, width).chunk(3, dim=2)
        x_B = self.norm_B(x_B)
        query_B, key_B, value_B = self.qkv_B(x_B).view(batch, n_head, head_dim * 3, height, width).chunk(3, dim=2)

        out_A = torch.einsum("bnchw, bncyx -> bnhwyx", query_B, key_A).contiguous() / math.sqrt(channel)
        out_A = out_A.view(batch, n_head, height * width, height * width)
        out_A = torch.softmax(out_A, dim=-1)
        out_A = out_A.view(batch, n_head, height, width, height, width)
        out_A = torch.einsum("bnhwyx, bncyx -> bnchw", out_A, value_A).contiguous()
        out_A = self.out_A(out_A.view(batch, channel, height, width))
        out_A = out_A + x_A

        out_B = torch.einsum("bnchw, bncyx -> bnhwyx", query_A, key_B).contiguous() / math.sqrt(channel)
        out_B = out_B.view(batch, n_head, height * width, height * width)
        out_B = torch.softmax(out_B, dim=-1)
        out_B = out_B.view(batch, n_head, height, width, height, width)
        out_B = torch.einsum("bnhwyx, bncyx -> bnchw", out_B, value_B).contiguous()
        out_B = self.out_B(out_B.view(batch, channel, height, width))
        out_B = out_B + x_B

        return out_A, out_B


class SMDHR_Block(nn.Module):
    def __init__(
            self, dim, num_heads, window_size, patch_size,
            sumdepth, depth, num_head, mlp_ratio, attn_ratio, num_layers=(2, 1), stage_depth=1, top_k_expert=3,
            top_k_trans=0.5, top_k_lr=0.5, top_k_cross=0.5, top_k_spa=0.5, top_k_spe=0.5, bias=False, use_dynamic_k=True
    ):
        super(SMDHR_Block, self).__init__()
        self.num_layers = num_layers
        self.use_dynamic_k = use_dynamic_k

        if self.num_layers[0] > 0:
            self.conv_spa_1 = nn.Conv2d(dim, dim, 3, 1, 1, bias=bias, groups=1)
        if self.num_layers[1] > 0:
            self.conv_spe_1 = nn.Conv2d(dim, dim, 3, 1, 1, bias=bias, groups=1)

        if self.num_layers[0] > 0 and self.num_layers[1] > 0:
            self.cross_att = Cross_attention(dim, norm_groups=dim // 4, top_k_ratio=top_k_cross,
                                             dynamic_k=use_dynamic_k)
        if self.num_layers[0] > 0 or self.num_layers[1] > 0:
            self.feature_fusion = Fusion_Embed(embed_dim=dim)

        self.prompt_guidance = FeatureWiseAffine(in_channels=512, out_channels=dim)

        if self.num_layers[0] > 0:
            self.TSST_branch = TSSTLayer(network_depth=sumdepth, dim=dim, depth=depth,
                                         num_heads=num_head, mlp_ratio=mlp_ratio,
                                         norm_layer=RLN, window_size=window_size,
                                         attn_ratio=attn_ratio, attn_loc='last', conv_type='DWConv',
                                         top_k_spa=top_k_spa, top_k_spe=top_k_spe, dynamic_k=use_dynamic_k)

        if self.num_layers[0] > 0 and self.num_layers[1] > 0:
            self.TSST_branch = TSSTLayer(network_depth=sumdepth, dim=dim, depth=depth,
                                         num_heads=num_head, mlp_ratio=mlp_ratio,
                                         norm_layer=RLN, window_size=window_size,
                                         attn_ratio=attn_ratio, attn_loc='last', conv_type='DWConv',
                                         top_k_spa=top_k_spa, top_k_spe=top_k_spe, dynamic_k=True)
            self.FRMoE_branch = FRMoELayer(dim, num_layers[1], stage_depth=stage_depth, top_kexp=top_k_expert,
                                           top_k_ratio=top_k_lr, dynamic_k=True)

    def forward(self, x, text_emb, k_ratios=None):
        if self.num_layers[0] > 0 and self.num_layers[1] > 0:  # EAFRB
            fea1 = self.prompt_guidance(self.conv_spa_1(x), text_emb)
            fea2 = self.prompt_guidance(self.conv_spe_1(x), text_emb)
            fea1 = self.TSST_branch(fea1, k_ratios[:, 0] if k_ratios is not None else None,
                                    k_ratios[:, 1] if k_ratios is not None else None)
            fea2 = self.FRMoE_branch(x, text_emb, fea2)

            # Use the Cross Attention present in the Variant 6 standard
            fea1, fea2 = self.cross_att(fea1, fea2, k_ratios[:, 0] if k_ratios is not None else None)

            x = self.feature_fusion(fea1, fea2)
        else:
            if self.num_layers[0] > 0:  # TSST
                x = self.prompt_guidance(self.conv_spa_1(x), text_emb)
                x = self.TSST_branch(x, k_ratios[:, 0] if k_ratios is not None else None,
                                     k_ratios[:, 1] if k_ratios is not None else None)
        return x


class SMDHR(nn.Module):
    def __init__(
            self, img_size=(128, 128), in_channel=305, embeding_dim=64, num_heads=8, window_size=(8, 8, 8),
            patch_size=(4, 4, 4),
            bias=False, LayerNorm_type="WithBias", mlp_ratios=[2., 4., 4., 2., 2.], stage_depth=1, top_k_expert=3,
            depths=[4, 4, 4, 2, 2], num_headr=[2, 4, 8, 1, 1], attn_ratio=[1 / 4, 1 / 2, 3 / 4, 0, 0],
            topk_ratio=[0.5, 0.5, 0.5, 0.5, 0.5],
            use_dynamic_k=True
    ):
        super(SMDHR, self).__init__()

        self.degradation_router = F_ext(in_nc=in_channel, nf=512)
        self.k_predictor = DynamicTopKPredictor(512)

        self.patch_embed = PatchEmbedmain(
            patch_size=1, in_chans=in_channel, embed_dim=embeding_dim, kernel_size=3)

        self.encoder0 = SMDHR_Block(
            embeding_dim, num_heads // 2, window_size[2], patch_size[2],
            sum(depths), depths[0], num_headr[0], mlp_ratios[0], attn_ratio[0], num_layers=(1, 0),
            stage_depth=stage_depth, top_k_expert=top_k_expert,
            top_k_trans=topk_ratio[0], top_k_lr=topk_ratio[1], top_k_cross=topk_ratio[2], top_k_spa=topk_ratio[3],
            top_k_spe=topk_ratio[4], bias=bias,
            use_dynamic_k=use_dynamic_k
        )
        self.patch_merge0 = PatchEmbedmain(
            patch_size=2, in_chans=embeding_dim, embed_dim=embeding_dim * 2 ** 1)
        self.encoder1 = SMDHR_Block(
            embeding_dim * 2 ** 1, num_heads // 2, window_size[2], patch_size[2],
            sum(depths), depths[0], num_headr[0], mlp_ratios[0], attn_ratio[0], num_layers=(1, 0),
            stage_depth=stage_depth, top_k_expert=top_k_expert,
            top_k_trans=topk_ratio[0], top_k_lr=topk_ratio[1], top_k_cross=topk_ratio[2], top_k_spa=topk_ratio[3],
            top_k_spe=topk_ratio[4], bias=bias,
            use_dynamic_k=use_dynamic_k
        )
        self.patch_merge1 = PatchEmbedmain(
            patch_size=2, in_chans=embeding_dim * 2 ** 1, embed_dim=embeding_dim * 2 ** 2)
        self.encoder2 = SMDHR_Block(
            embeding_dim * 2 ** 2, num_heads, window_size[1], patch_size[1],
            sum(depths), depths[1], num_headr[1], mlp_ratios[1], attn_ratio[1], num_layers=(1, 1),
            stage_depth=stage_depth, top_k_expert=top_k_expert,
            top_k_trans=topk_ratio[0], top_k_lr=topk_ratio[1], top_k_cross=topk_ratio[2], top_k_spa=topk_ratio[3],
            top_k_spe=topk_ratio[4], bias=bias,
            use_dynamic_k=use_dynamic_k
        )
        self.patch_merge2 = PatchEmbedmain(
            patch_size=2, in_chans=embeding_dim * 2 ** 2, embed_dim=embeding_dim * 2 ** 2)
        self.mid = SMDHR_Block(
            embeding_dim * 2 ** 2, num_heads, window_size[0], patch_size[0],
            sum(depths), depths[2], num_headr[2], mlp_ratios[2], attn_ratio[2], num_layers=(1, 2),
            stage_depth=stage_depth, top_k_expert=top_k_expert,
            top_k_trans=topk_ratio[0], top_k_lr=topk_ratio[1], top_k_cross=topk_ratio[2], top_k_spa=topk_ratio[3],
            top_k_spe=topk_ratio[4], bias=bias,
            use_dynamic_k=use_dynamic_k
        )
        self.patch_split1 = PatchUnEmbedmain(
            patch_size=2, out_chans=embeding_dim * 2 ** 2, embed_dim=embeding_dim * 2 ** 2)
        self.decoder1 = SMDHR_Block(
            embeding_dim * 2 ** 2, num_heads, window_size[1], patch_size[1],
            sum(depths), depths[1], num_headr[1], mlp_ratios[1], attn_ratio[1], num_layers=(1,
                                                                                            1), stage_depth=stage_depth,
            top_k_expert=top_k_expert,
            top_k_trans=topk_ratio[0], top_k_lr=topk_ratio[1], top_k_cross=topk_ratio[2], top_k_spa=topk_ratio[3],
            top_k_spe=topk_ratio[4], bias=bias,
            use_dynamic_k=use_dynamic_k
        )
        self.patch_split2 = PatchUnEmbedmain(
            patch_size=2, out_chans=embeding_dim * 2 ** 1, embed_dim=embeding_dim * 2 ** 2)
        self.decoder2 = SMDHR_Block(
            embeding_dim * 2 ** 1, num_heads // 2, window_size[2], patch_size[2],
            sum(depths), depths[0], num_headr[0], mlp_ratios[0], attn_ratio[0], num_layers=(1, 0),
            stage_depth=stage_depth, top_k_expert=top_k_expert,
            top_k_trans=topk_ratio[0], top_k_lr=topk_ratio[1], top_k_cross=topk_ratio[2], top_k_spa=topk_ratio[3],
            top_k_spe=topk_ratio[4], bias=bias,
            use_dynamic_k=use_dynamic_k
        )
        self.patch_split3 = PatchUnEmbedmain(
            patch_size=2, out_chans=embeding_dim, embed_dim=embeding_dim * 2 ** 1)
        self.decoder3 = SMDHR_Block(
            embeding_dim, num_heads // 2, window_size[2], patch_size[2],
            sum(depths), depths[0], num_headr[0], mlp_ratios[0], attn_ratio[0], num_layers=(1, 0),
            stage_depth=stage_depth, top_k_expert=top_k_expert,
            top_k_trans=topk_ratio[0], top_k_lr=topk_ratio[1], top_k_cross=topk_ratio[2], top_k_spa=topk_ratio[3],
            top_k_spe=topk_ratio[4], bias=bias,
            use_dynamic_k=use_dynamic_k
        )
        self.patch_unembed = PatchUnEmbedmain(
            patch_size=1, out_chans=in_channel, embed_dim=embeding_dim, kernel_size=3)

        self.skip0 = nn.Conv2d(embeding_dim, embeding_dim, 1)
        self.skip1 = nn.Conv2d(embeding_dim * 2 ** 1, embeding_dim * 2 ** 1, 1)
        self.skip2 = nn.Conv2d(embeding_dim * 2 ** 2, embeding_dim * 2 ** 2, 1)

        self.fusion1 = SKFusion(embeding_dim * 2 ** 2)
        self.fusion2 = SKFusion(embeding_dim * 2 ** 1)
        self.fusion3 = SKFusion(embeding_dim)

        self.L1Loss = torch.nn.L1Loss()
        self.sam_loss = SAMLoss()
        self.BandWiseMSE = BandWiseMSE()
        self.Waveletloss = HyperspectralSWTLoss()

    # Variant 6: modify the forward signature to stay compatible with the original (ori) calling convention
    def forward(self, x, ori, x_gt=None):
        deg_embed = self.degradation_router(x)
        x = self.patch_embed(x)
        emb = deg_embed
        k_ratios = self.k_predictor(emb)

        x = self.encoder0(x, emb, k_ratios)
        skip0 = x

        x = self.patch_merge0(x)
        x = self.encoder1(x, emb, k_ratios)
        skip1 = x

        x = self.patch_merge1(x)
        x = self.encoder2(x, emb, k_ratios)
        skip2 = x

        x = self.patch_merge2(x)
        x = self.mid(x, emb, k_ratios)
        x = self.patch_split1(x)

        x = self.fusion1([x, self.skip2(skip2)]) + x
        x = self.decoder1(x, emb, k_ratios)
        x = self.patch_split2(x)

        x = self.fusion2([x, self.skip1(skip1)]) + x
        x = self.decoder2(x, emb, k_ratios)
        x = self.patch_split3(x)

        x = self.fusion3([x, self.skip0(skip0)]) + x
        x = self.decoder3(x, emb, k_ratios)
        x = self.patch_unembed(x)

        if x_gt is not None:
            loss1 = torch.unsqueeze(self.L1Loss(x, x_gt), 0)
            loss2 = torch.unsqueeze(self.BandWiseMSE(x, x_gt), 0)
            loss3 = torch.unsqueeze(self.sam_loss(x, x_gt), 0)
            loss4 = torch.unsqueeze(self.Waveletloss(x, x_gt), 0)
            return x, loss1, loss2, loss3, loss4
        else:
            return x


def SMDHR_b():
    return SMDHR(embeding_dim=48, attn_ratio=[1 / 4, 1 / 2, 3 / 4, 1 / 2, 1 / 4])

if __name__ == '__main__':
    from thop import profile
    import numpy as np
    import time
    net = SMDHR_b().to("cuda" if torch.cuda.is_available() else "cpu")
    input = torch.randn(1, 305, 256, 256).to("cuda" if torch.cuda.is_available() else "cpu")
    ts = time.time()
    out = net(input, input)
    te = time.time()
    print('Time = ' + str(te - ts) + 'S')
    macs, _ = profile(net, inputs=(input, input, ))
    total = sum([param.nelement() for param in net.parameters()])
    print('Macs = ' + str(macs / 1000 ** 3) + 'G')
    print('Params = ' + str(total / 1e6) + 'M')