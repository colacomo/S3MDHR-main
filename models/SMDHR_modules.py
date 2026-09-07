from collections import OrderedDict
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import random
import numbers
import numpy as np
from models.COR_arch_util import DWConv, LayerNorm, Itv_concat, SAM, conv_block,SAM1,SAM2
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.distributions.normal import Normal
from utils.lossfunc import HyperspectralSWTLoss, SAMLoss, BandWiseMSE
from models.PHSImodule import (
    LayerNorm,
    Fusion_Embed,
    FeatureWiseAffine,
    initialize_weights
)
from torch.nn import init as init

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

class MySequential(nn.Sequential):
    def forward(self, x1, x2):
        for layer in self:
            if isinstance(layer, nn.Module):
                x1 = layer(x1, x2)
            else:
                x1 = layer(x1, x2)
        return x1

def softmax_with_temperature(logits, temperature=1.0):
    scaled_logits = logits / temperature
    return F.softmax(scaled_logits, dim=-1)

class SparseDispatcher(object):
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts

        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates.unsqueeze(-1).unsqueeze(-1))
        zeros = torch.zeros(
            self._gates.size(0),
            expert_out[-1].size(1),
            expert_out[-1].size(2),
            expert_out[-1].size(3),
            requires_grad=True,
            device=stitched.device
        )
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined

    def to_spatial(self, x, x_shape):
        h, w = x_shape
        amp, phase = x.chunk(2, dim=1)
        real = amp * torch.cos(phase)
        imag = amp * torch.sin(phase)
        x = real + 1j * imag
        x = torch.fft.ifft2(x, s=(h, w), norm="backward").real
        return x

    def expert_to_gates(self):
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

class Clean_route(nn.Module):
    def __init__(self,dim):
        super(Clean_route,self).__init__()
    def forward(self,img_degra,degra_map):
        return img_degra

class Denoise_route(nn.Module):
    def __init__(self,dim):
        super(Denoise_route,self).__init__()
        self.dim = dim
        self.noise = nn.Sequential(
            default_conv(2*dim, dim, 3),
            default_conv(dim, dim // 8, 3),
            nn.SELU(),
            default_conv(dim // 8, dim, 3),
            nn.SELU()
        )
    def forward(self,img_degra,degra_map):
        noise = self.noise(degra_map)
        img_degra = noise+img_degra
        return img_degra

class Dehaze_route(nn.Module):
    def __init__(self,dim):
        super(Dehaze_route,self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
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
    channels = feat_in.size(1)
    N, kernels, H, W = kernel.size()
    pad_sz = (ksize - 1) // 2

    feat_in = F.pad(feat_in, (pad_sz, pad_sz, pad_sz, pad_sz), mode="replicate")
    feat_in = feat_in.unfold(2, ksize, 1).unfold(3, ksize, 1)
    feat_in = feat_in.permute(0, 2, 3, 1, 5, 4).contiguous()
    feat_in = feat_in.reshape(N, H, W, channels, -1)

    kernel = kernel.permute(0, 2, 3, 1).reshape(N, H, W, channels, ksize, ksize)
    kernel = kernel.permute(0, 1, 2, 3, 5, 4).reshape(N, H, W, channels, -1)
    feat_out = torch.sum(feat_in * kernel, axis=-1)
    feat_out = feat_out.permute(0, 3, 1, 2).contiguous()
    return feat_out

class Deblur_route(nn.Module):
    def __init__(self,dim,kpn_sz=5):
        super(Deblur_route,self).__init__()
        self.kpn_sz = kpn_sz
        self.convolve = nn.Sequential(
            default_conv(2*dim, dim, 3),
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

class DenseBlock_5C(nn.Module):
    def __init__(self, nf, gc=32, bias=False, groups=4, squeeze_ratio=16, memory_blocks=64, top_k_ratio=0.5,
                 dynamic_k=False):
        super(DenseBlock_5C, self).__init__()
        self.squeeze_ratio = squeeze_ratio
        self.memory_blocks = memory_blocks
        self.dynamic_k = dynamic_k
        self.top_k_ratio = top_k_ratio if not dynamic_k else None

        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias, groups=groups)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias, groups=groups)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias, groups=groups)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias, groups=groups)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias, groups=groups)

        self.lrelu = nn.LeakyReLU(0.2, True)

        initialize_weights(
            [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5],
            0.1
        )

    def forward(self, x, k_ratio=None):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x5

class ModExpert(nn.Module):
    def __init__(self, dim:int, func: nn.Module, depth=1, top_k_ratio=0, dynamic_k=True):
        super(ModExpert, self).__init__()
        self.depth = depth
        self.body = func

    def process(self, x, degra_map):
        x = self.body(x, degra_map)
        return x

    def feat_extract(self, feats, degra_map):
        for _ in range(self.depth):
            feat = self.process(feats, degra_map)
        return feat

    def forward(self, x, degra_map):
        b, c, h, w = x.shape
        if b == 0:
            return x
        else:
            x = self.feat_extract(x, degra_map)
            return x

class AdapterLayer(nn.Module):
    def __init__(self,
                 dim: int,
                 num_experts: int = 10,
                 top_k: int = 3,
                 stage_depth: int = 1,
                 freq_dim: int = 128,
                 with_complexity: bool = False,
                 complexity_scale: str = "min"
                 ):
        super().__init__()
        self.tau = 1
        self.loss = None
        self.top_k = top_k
        self.noise_eps = 1e-2
        self.num_experts = num_experts

        self.proj = DenseBlock_5C(dim)
        expert_layer1_1 = Denoise_route(dim=dim)
        expert_layer1_2 = Denoise_route(dim=dim)
        expert_layer1_3 = Denoise_route(dim=dim)
        expert_layer2_1 = Dehaze_route(dim=dim)
        expert_layer2_2 = Dehaze_route(dim=dim)
        expert_layer2_3 = Dehaze_route(dim=dim)
        expert_layer3_1 = Deblur_route(dim=dim, kpn_sz=3)
        expert_layer3_2 = Deblur_route(dim=dim, kpn_sz=5)
        expert_layer3_3 = Deblur_route(dim=dim, kpn_sz=7)
        expert_layer4 = Clean_route(dim=dim)
        self.experts = nn.ModuleList([
            ModExpert(dim, func=expert_layer1_1, depth=stage_depth),
            ModExpert(dim, func=expert_layer1_2, depth=stage_depth),
            ModExpert(dim, func=expert_layer1_3, depth=stage_depth),
            ModExpert(dim, func=expert_layer2_1, depth=stage_depth),
            ModExpert(dim, func=expert_layer2_2, depth=stage_depth),
            ModExpert(dim, func=expert_layer2_3, depth=stage_depth),
            ModExpert(dim, func=expert_layer3_1, depth=stage_depth),
            ModExpert(dim, func=expert_layer3_2, depth=stage_depth),
            ModExpert(dim, func=expert_layer3_3, depth=stage_depth),
            ModExpert(dim, func=expert_layer4, depth=stage_depth),
        ])

        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=False)

        expert_complexity = torch.tensor([
            sum(p.numel() for p in expert.parameters())
            for expert in self.experts
        ])

        self.routing = RoutingFunction(
            dim, freq_dim,
            num_experts=num_experts,
            k=top_k,
            complexity=expert_complexity,
            use_complexity_bias=with_complexity,
            complexity_scale=complexity_scale
        )

    def forward(self, x, freq_emb, degra_map):
        gates, top_k_indices, top_k_values = self.routing(x, freq_emb)
        degra_map = self.proj(degra_map)
        degra_map = Itv_concat(x, degra_map)

        if self.training:
            dispatcher = SparseDispatcher(self.num_experts, gates)
            expert_inputs = dispatcher.dispatch(x)
            expert_inputs2 = dispatcher.dispatch(degra_map)
            expert_outputs = [
                self.experts[i](expert_inputs[i], expert_inputs2[i])
                for i in range(self.num_experts)
            ]
            out = dispatcher.combine(expert_outputs, multiply_by_gates=True)
        else:
            selected_experts = [self.experts[i] for i in top_k_indices.squeeze(0)]
            expert_outputs = torch.stack(
                [expert(x, degra_map) for expert in selected_experts],
                dim=1
            )
            gates_topk = gates.gather(1, top_k_indices)
            weighted_outputs = gates_topk.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * expert_outputs
            out = weighted_outputs.sum(dim=1)

        out = self.proj_out(out)
        return out

class RoutingFunction(nn.Module):
    def __init__(self, dim, freq_dim, num_experts, k, complexity, use_complexity_bias=True, complexity_scale="max"):
        super(RoutingFunction, self).__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('b c 1 1 -> b c'),
            nn.Linear(dim, num_experts, bias=False)
        )
        self.freq_gate = nn.Linear(freq_dim, num_experts, bias=False)

        if complexity_scale == "min":
            complexity = complexity / complexity.min()
        elif complexity_scale == "max":
            complexity = complexity / complexity.max()
        self.register_buffer('complexity', complexity)

        self.k = k
        self.tau = 1
        self.num_experts = num_experts
        self.noise_std = (1.0 / num_experts) * 1.0
        self.use_complexity_bias = use_complexity_bias

    def forward(self, x, freq_emb):
        logits = self.gate(x) + self.freq_gate(freq_emb)
        noise = torch.randn_like(logits) * self.noise_std
        noisy_logits = logits + noise
        gating_scores = noisy_logits.softmax(dim=-1)
        top_k_values, top_k_indices = torch.topk(gating_scores, self.k, dim=-1)
        gates = torch.zeros_like(logits).scatter_(1, top_k_indices, top_k_values)
        return gates, top_k_indices, top_k_values

    def importance_loss(self, gating_scores):
        importance = gating_scores.sum(dim=0)
        if self.use_complexity_bias:
            importance = importance * (self.complexity * self.tau)
        imp_mean = importance.mean()
        imp_std = importance.std()
        loss_imp = (imp_std / (imp_mean + 1e-8)) ** 2
        return loss_imp

    def load_loss(self, logits, logits_noisy, noise_std):
        thresholds = torch.topk(logits_noisy, self.k, dim=-1).indices[:, -1]
        threshold_per_item = torch.sum(
            F.one_hot(thresholds, self.num_experts) * logits_noisy,
            dim=-1
        )
        noise_required_to_win = (threshold_per_item.unsqueeze(-1) - logits) / noise_std
        normal_dist = Normal(0, 1)
        p = 1.0 - normal_dist.cdf(noise_required_to_win)
        p_mean = p.mean(dim=0)
        p_mean_std = p_mean.std()
        p_mean_mean = p_mean.mean()
        loss_load = (p_mean_std / (p_mean_mean + 1e-8)) ** 2
        return loss_load