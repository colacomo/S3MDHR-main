import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from models.utils.arch_util import DWConv, LayerNorm, Itv_concat, SAM, SAM1, conv_block
from models.utils.transformerBLC_util import Standard_Mlp
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import logging

logger = logging.getLogger('base')


class FeedForward(nn.Module):
    """
    x: B,C,H,W
    return: B,C,H,W
    process: 1x1 conv + 3x3 dwconv + gate + 1x1 conv
    Adopted: Restormer —— Gated-Dconv Feed-Forward Network (GDFN)
    """
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class DW_Mlp(nn.Module):
    """
    x: B,hw,C
    return: B,hw,C
    process: mlp + 3x3 dwconv + gelu(drop) + mlp(drop)
    Adopted: Transweather
    """
    def __init__(self, dim, ffn_expansion_factor, bias, act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.fc1 = nn.Conv2d(dim, hidden_features, 1, 1, 0, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, dim, 1, 1, 0, bias=bias)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class Key_Attention(nn.Module):
    """
    x: B,C,H,W   key: B,L,C
    return: B,C,H,W
    process: 1x1 qkv_conv + 3x3 dwconv + normalize(hw) + cxc attention(temperature) -> value + 1x1 conv
    Adopted: Restormer —— Multi-DConv Head Transposed Self-Attention (MDTA)
    ps: attention(scale) == normalize + attention
    """
    def __init__(self, dim, dimkey, num_heads, bias,qk_scale=None,attn_drop=0.,proj_drop=0.):
        super(Key_Attention, self).__init__()
        self.num_heads = num_heads
        self.scale = qk_scale or dim ** -0.5
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        )
        self.kv = nn.Linear(dimkey, dim*2)

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, key):
        b, c, h, w = x.shape

        q = self.q(x)
        kv = self.kv(key)
        k, v = kv.chunk(2, dim=2)

        q = rearrange(q, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        k = rearrange(k, 'b k (head c) -> b head k c', head=self.num_heads)
        v = rearrange(v, 'b k (head c) -> b head k c', head=self.num_heads)


        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v)

        out = rearrange(out, 'b head (h w) c -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        out = self.proj_drop(out)
        return out


class TransformerBlock_4DFTB(nn.Module):
    """
    x: B,C,H,W   key: B,K,C
    return: B,C,H,W
    process: MDTA + GDFN
    params: dim, num_heads, ffn_expansion_factor, bias:True/false LayerNorm_type: BiasFree/WithBias
    Adopted:: Restormer
    """
    def __init__(self, dim, dimkey, num_heads, ffn_expansion_factor, bias, LayerNorm_type, principle=True, sam=False, ops_type=4, pred=False):
        super(TransformerBlock_4DFTB, self).__init__()

        self.normkey = nn.LayerNorm(dimkey, elementwise_affine=False)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Key_Attention(dim, dimkey, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = DW_Mlp(dim, ffn_expansion_factor, bias)
        self.sam = sam
        self.principle = principle
        if principle:
            self.principle = MPB(dim=dim, ops_type=ops_type,pred=pred)
        if sam:
            self.SAM = SAM(n_feat=dim, kernel_size=1, bias=bias)

    def forward(self, im_degra, key, prompt, resize_img=None,degra_type=None):
        prompt = prompt.unsqueeze(dim=2).unsqueeze(dim=3)
        pred = self.principle.pred(prompt)
        if degra_type is None:
            key0, key1, key2 = key.shape
            degra_key = key.reshape(key0, key1 * key2)
            degra_key = torch.squeeze(pred) @ degra_key
            key = degra_key.reshape(-1, key1, key2)

        if self.sam:
            degra_map, img = self.SAM(im_degra,resize_img)
            degra_map = self.attn(self.norm1(degra_map), self.normkey(key))
        else:
            degra_map = self.attn(self.norm1(im_degra), self.normkey(key))

        if self.principle:
            im_degra, pred = self.principle(im_degra,degra_map,prompt,degra_type=degra_type)
        else:
            im_degra = im_degra - degra_map*im_degra

        im_degra = im_degra + self.ffn(self.norm2(im_degra))

        if self.sam:
            return im_degra, img, pred
        else:
            return im_degra, resize_img, pred



class MPB(nn.Module):
    def __init__(self,dim,pred=False,ops_type=6):
        super(MPB,self).__init__()
        self.mlp_img = nn.Conv2d(dim,dim,1,1,0)
        self.mlp_degra = nn.Conv2d(dim,dim,1,1,0)
        self.deblur = Deblur_route(dim=dim)
        self.denoise = Denoise_route(dim=dim)
        self.dehaze = Dehaze_route(dim=dim)
        self.dedark = Dedark_route(dim=dim)
        self.identity = Identity_route()
        self.de_dict = {'deblur': self.deblur, 'denoise': self.denoise, 'dehaze': self.dehaze, 'dedark': self.dedark, 'clean': self.identity}
        self.flag = pred
        if pred:
            self.pred = nn.Sequential(
                nn.Conv2d(dim,dim,1,1,0),
                nn.Conv2d(dim,ops_type,1,1,0),
                nn.Softmax(dim=1)
            )

    def forward(self,img,degra_map,prompt,degra_type=None):
        b,c,h,w = img.shape
        pred = self.pred(prompt)    #    B,C,H,W -> B,K,1,1

        img = self.mlp_img(img)
        degra_map = self.mlp_degra(degra_map)
        degra_map = Itv_concat(img,degra_map)
        if degra_type is not None:
            # stage 1 training
            fn = self.de_dict[degra_type]
            out = fn(img,degra_map)
            return out, pred
        else:
            # stage 2 training
            out_deblur = self.deblur(img, degra_map)
            out_denoise = self.denoise(img, degra_map)
            out_dehaze = self.dehaze(img, degra_map)
            out_dedark = self.dedark(img, degra_map)

            weight_deblur = pred[:, 0, :, :]
            weight_denoise = pred[:, 1, :, :]
            weight_dehaze = pred[:, 2, :, :]
            weight_dedark = pred[:, 3, :, :]

            out = weight_deblur.unsqueeze(-1)*out_deblur + weight_denoise.unsqueeze(-1)*out_denoise + weight_dehaze.unsqueeze(-1)*out_dehaze + weight_dedark.unsqueeze(-1)*out_dedark
            return out, pred





class Identity_route(nn.Module):
    def __init__(self):
        super(Identity_route,self).__init__()
    def forward(self,img_degra,degra_map):
        return img_degra


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

class Denoise_route(nn.Module):
    def __init__(self,dim):
        super(Denoise_route,self).__init__()
        self.dim = dim
        self.noise = nn.Sequential(
            default_conv(dim, dim, 3),
            default_conv(dim, dim // 8, 3),
            nn.SELU(),
            default_conv(dim // 8, dim, 3),
            nn.SELU()
        )
    def forward(self,img_degra,degra_map):
        _, noise = torch.split(degra_map, (self.dim, self.dim), dim=1)
        noise = self.noise(noise)
        img_degra = noise+img_degra
        return img_degra

class Dedark_route(nn.Module):
    def __init__(self,dim):
        super(Dedark_route,self).__init__()
        self.dim = dim
        self.mul = nn.Sequential(
            default_conv(dim, dim, 3),
            default_conv(dim, dim // 8, 3),
            nn.SELU(),
            default_conv(dim // 8, dim, 3),
            nn.SELU()
        )
        self.add = nn.Sequential(
            default_conv(dim, dim, 3),
            default_conv(dim, dim // 8, 3),
            nn.SELU(),
            default_conv(dim // 8, dim, 3),
            nn.SELU()
        )

    def forward(self, img_degra, degra_map):
        mul, add = torch.split(degra_map, (self.dim, self.dim), dim=1)
        mul = self.mul(mul)
        add = self.add(add)
        img_degra = torch.mul(img_degra, mul) + add
        return img_degra

class Dehaze_route(nn.Module):
    def __init__(self,dim):
        super(Dehaze_route,self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)#max
        self.a = nn.Sequential(
            default_conv(dim, dim // 8, 3),
            nn.SELU(),
            default_conv(dim // 8, dim, 3),
            nn.SELU()
        )
        self.t = nn.Sequential(
            default_conv(dim, dim, 3),
            default_conv(dim, dim // 8, 3),
            nn.SELU(),
            default_conv(dim // 8, dim, 3),
            nn.SELU()
        )

    def forward(self, img_degra, degra_map):
        t, a = torch.split(degra_map, (self.dim, self.dim), dim=1)
        a = self.a(self.avg_pool(a))
        t = self.t(t)
        img_degra = torch.mul(t, (img_degra - a)) + a
        return img_degra


def kernel2d_conv(feat_in, kernel, ksize):
    """
    If you have some problems in installing the CUDA FAC layer,
    you can consider replacing it with this Python implementation.
    Thanks @AIWalker-Happy for his implementation.
    """
    channels = feat_in.size(1)
    N, kernels, H, W = kernel.size()
    pad_sz = (ksize - 1) // 2

    feat_in = F.pad(feat_in, (pad_sz, pad_sz, pad_sz, pad_sz), mode="replicate")  # B,C,H+k-1,W+k-1
    feat_in = feat_in.unfold(2, ksize, 1).unfold(3, ksize, 1)     # B,C,H,W,k,k  sliding-window op of size ksize and stride 1 over dims 2 and 3
    feat_in = feat_in.permute(0, 2, 3, 1, 5, 4).contiguous()      # B,H,W,C,k,k  B,C,H,W,k,k -> B,H,W,C,k,k
    feat_in = feat_in.reshape(N, H, W, channels, -1)              # B,H,W,C,k*k  B,H,W,C,k,k -> B,H,W,C,k*k

    kernel = kernel.permute(0, 2, 3, 1).reshape(N, H, W, channels, ksize, ksize)  # B,C*k*k,H,W -> B,H,W,C*k*k -> B,H,W,C,k,k
    kernel = kernel.permute(0, 1, 2, 3, 5, 4).reshape(N, H, W, channels, -1)      # B,H,W,C,k,k -> B,H,W,C,k*k
    feat_out = torch.sum(feat_in * kernel, axis=-1)               # B,H,W,C
    feat_out = feat_out.permute(0, 3, 1, 2).contiguous()          # B,C,H,W
    return feat_out


class Deblur_route(nn.Module):
    def __init__(self,dim,kpn_sz=5):
        super(Deblur_route,self).__init__()
        self.kpn_sz = kpn_sz
        self.convolve = nn.Sequential(
            default_conv(dim*2, dim, 3),
            default_conv(dim, dim // 8, 3),
            nn.SELU(),
            default_conv(dim // 8, dim, 3),
            nn.SELU(),
            conv_block(dim, dim * (kpn_sz ** 2), kernel_size=1),
        )
    def forward(self,img_degra,degra_map):
        blur_kernel = self.convolve(degra_map)
        img_degra = kernel2d_conv(img_degra,blur_kernel,self.kpn_sz)
        return img_degra


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x










