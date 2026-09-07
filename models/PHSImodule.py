import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numbers
import math
from einops import rearrange

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming init for conv layers (fan_in mode)
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                # Scale weights for residual connections
                m.weight.data *= scale
                if m.bias is not None:
                    # Zero the bias
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                # Kaiming init for linear layers
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                # Init batch norm weight to 1 and bias to 0
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

class PromptAdapter(nn.Module):
    def __init__(self, in_dim, act=nn.ReLU(), bias=False):
        super(PromptAdapter, self).__init__()
        # Down-projection linear layer
        self.linear_dw = nn.Linear(in_dim, in_dim // 8, bias=bias)
        # Activation function
        self.act = act
        # Up-projection linear layer
        self.linear_up = nn.Linear(in_dim // 8, in_dim, bias=bias)
        # Layer norm
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x):
        # Residual connection
        res = x
        # Down-project
        x = self.linear_dw(x)
        # Activate
        x = self.act(x)
        # Up-project
        x = self.linear_up(x)
        # Normalize and add the residual
        x = self.act(self.norm(x) + res)
        return x

#################################### Text-IF #######################################

## Feature Modulation
class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=True):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        # MLP generates the affine parameters
        self.MLP = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.LeakyReLU(),
            nn.Linear(in_channels * 2, out_channels * (1 + self.use_affine_level)),
        )
        # Adapter for the text embedding
        self.adapter = PromptAdapter(512, act=nn.LeakyReLU(), bias=True)

    def forward(self, x, text_embed):
        # Adapt the text embedding
        text_embed = self.adapter(text_embed)
        batch = x.shape[0]
        if self.use_affine_level:
            # Generate the affine parameters gamma and beta
            gamma, beta = self.MLP(text_embed).view(batch, -1, 1, 1).chunk(2, dim=1)
            # Apply affine transform: x = (1 + gamma) * x + beta
            x = (1 + gamma) * x + beta
        return x

class Fusion_Embed(nn.Module):
    def __init__(self, embed_dim, bias=False):
        super(Fusion_Embed, self).__init__()
        # 1x1 conv to fuse the two input features
        self.fusion_proj = nn.Conv2d(
            embed_dim * 2, embed_dim, kernel_size=1, stride=1, bias=bias
        )

    def forward(self, x_A, x_B):
        # Concatenate the two features along the channel dimension
        x = torch.concat([x_A, x_B], dim=1)
        # Fuse with a 1x1 conv
        x = self.fusion_proj(x)
        return x

class Attention_spatial(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=16):
        super().__init__()
        self.n_head = n_head
        # Group norm
        self.norm = nn.GroupNorm(norm_groups, in_channel)
        # 1x1 conv to generate Q, K, V
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        # Output projection
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head
        # Generate Q, K, V
        qkv = self.qkv(self.norm(input)).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)
        # Compute attention scores
        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        # Normalize attention weights
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)
        # Apply attention
        attn = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        # Output projection and residual connection
        attn = self.out(attn.view(batch, channel, height, width))
        return attn + input

class Key_Attention(nn.Module):
    """
    Simplified attention module adapted to the given input dimensions.
    x: B, C, H, W   (4, 48, 128, 128)
    key: B, D       (4, 512)
    return: B, C, H, W
    """

    def __init__(self, dim, dimkey, num_heads=8, bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(Key_Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Query branch for visual features
        self.q = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        )

        # Key/value branch for text features
        self.kv = nn.Linear(dimkey, dim * 2)  # outputs key and value

        # Output projection
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, key):
        b, c, h, w = x.shape

        # Process query (visual features)
        q = self.q(x)  # [4, 48, 128, 128]

        # Process key/value (text features)
        kv = self.kv(key)  # [4, 96] (48*2)
        k, v = kv.chunk(2, dim=1)  # each [4, 48]

        # Reshape into multi-head format
        q = rearrange(q, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        k = rearrange(k, 'b (head c) -> b head 1 c', head=self.num_heads)  # add sequence dimension
        v = rearrange(v, 'b (head c) -> b head 1 c', head=self.num_heads)  # add sequence dimension

        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention
        out = (attn @ v)
        out = rearrange(out, 'b head (h w) c -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # Output projection
        out = self.project_out(out)
        out = self.proj_drop(out)
        return out


class FeatureCA(nn.Module):
    def __init__(self, in_channels, out_channels, dimkey=512, num_heads=8, bias=True,
                 use_affine_level=True):
        """
        Modified FeatureWiseAffine module adapted to the input dimensions.

        Parameters:
            in_channels: 48 (number of channels in the input feature map)
            out_channels: 48 (number of channels in the output feature map)
            dimkey: 512 (text embedding dimension)
            num_heads: 8 (number of attention heads; 48 is divisible by 8)
            use_affine_level: whether to apply an additional affine transform
        """
        super(FeatureCA, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_affine_level = use_affine_level
        self.dimkey = dimkey

        # Text embedding adapter - keeps the 512 dimensions unchanged
        self.adapter = PromptAdapter(dimkey, act=nn.LeakyReLU(), bias=True)

        # Attention module
        self.attn = Key_Attention(
            dim=in_channels,
            dimkey=dimkey,
            num_heads=num_heads,
            bias=bias
        )

        # Layer norm - GroupNorm-style normalization over the channels
        self.norm_x = LayerNorm(in_channels, 'WithBias')  # equivalent to LayerNorm over the channels
        self.norm_key = nn.LayerNorm(dimkey)

        # Optional affine transform layer
        if self.use_affine_level:
            self.affine = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )
        else:
            # If no affine transform is used, make sure the output channels match
            if in_channels != out_channels:
                self.affine = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            else:
                self.affine = nn.Identity()

    def forward(self, x, text_embed):
        """
        Forward pass.

        Parameters:
            x: visual feature [4, 48, 128, 128]
            text_embed: text embedding [4, 512]

        Returns:
            modulated feature [4, 48, 128, 128]
        """
        # 1. Adapt the text embedding (stays at 512 dims)
        text_embed = self.adapter(text_embed)  # [4, 512]

        # 2. Normalize features
        x_norm = self.norm_x(x)  # [4, 48, 128, 128]
        key_norm = self.norm_key(text_embed)  # [4, 512]

        # 3. Apply attention
        attn_out = self.attn(x_norm, key_norm)  # [4, 48, 128, 128]

        # 4. Residual connection
        modulated_x = attn_out  # [4, 48, 128, 128]

        # 5. Apply the affine transform
        output = self.affine(modulated_x)  # [4, 48, 128, 128]

        return output

class Cross_attention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=16):
        super().__init__()
        self.n_head = n_head
        # Normalization for both branches
        self.norm_A = nn.GroupNorm(norm_groups, in_channel)
        self.norm_B = nn.GroupNorm(norm_groups, in_channel)
        # Generate Q, K, V for branch A
        self.qkv_A = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out_A = nn.Conv2d(in_channel, in_channel, 1)
        # Generate Q, K, V for branch B
        self.qkv_B = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out_B = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, x_A, x_B):
        batch, channel, height, width = x_A.shape
        n_head = self.n_head
        head_dim = channel // n_head

        # Process branch A
        x_A = self.norm_A(x_A)
        query_A, key_A, value_A = self.qkv_A(x_A).view(batch, n_head, head_dim * 3, height, width).chunk(3, dim=2)

        # Process branch B
        x_B = self.norm_B(x_B)
        query_B, key_B, value_B = self.qkv_B(x_B).view(batch, n_head, head_dim * 3, height, width).chunk(3, dim=2)

        # Compute cross-attention for branch A (using B's query)
        out_A = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query_B, key_A
        ).contiguous() / math.sqrt(channel)
        out_A = out_A.view(batch, n_head, height, width, -1)
        out_A = torch.softmax(out_A, -1)
        out_A = out_A.view(batch, n_head, height, width, height, width)
        out_A = torch.einsum("bnhwyx, bncyx -> bnchw", out_A, value_A).contiguous()
        out_A = self.out_A(out_A.view(batch, channel, height, width))
        out_A = out_A + x_A

        # Compute cross-attention for branch B (using A's query)
        out_B = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query_A, key_B
        ).contiguous() / math.sqrt(channel)
        out_B = out_B.view(batch, n_head, height, width, -1)
        out_B = torch.softmax(out_B, -1)
        out_B = out_B.view(batch, n_head, height, width, height, width)
        out_B = torch.einsum("bnhwyx, bncyx -> bnchw", out_B, value_B).contiguous()
        out_B = self.out_B(out_B.view(batch, channel, height, width))
        out_B = out_B + x_B

        return out_A, out_B

class Lightweight_Cross_attention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=16):
        super().__init__()
        self.n_head = n_head
        self.head_dim = in_channel // n_head

        # Shared normalization layer
        self.norm = nn.GroupNorm(norm_groups, in_channel)

        # Shared QK generation layer producing Q and K
        self.qk = nn.Conv2d(in_channel, in_channel * 2, 1, bias=False)  # generates Q and K
        self.out = nn.Conv2d(in_channel, in_channel, 1, bias=False)

    def forward(self, x_A, x_B):
        batch, channel, height, width = x_A.shape

        # Shared normalization
        x_A = self.norm(x_A)
        x_B = self.norm(x_B)

        # Generate Q and K
        qk_A = self.qk(x_A)
        qk_B = self.qk(x_B)
        query_A, key_A = qk_A[:, :channel, :, :], qk_A[:, channel:, :, :]
        query_B, key_B = qk_B[:, :channel, :, :], qk_B[:, channel:, :, :]

        # Use the raw input as V
        value_A = x_A
        value_B = x_B

        # Reshape into multi-head form
        query_A = query_A.view(batch, self.n_head, self.head_dim, height, width)
        key_A = key_A.view(batch, self.n_head, self.head_dim, height, width)
        value_A = value_A.view(batch, self.n_head, self.head_dim, height, width)
        query_B = query_B.view(batch, self.n_head, self.head_dim, height, width)
        key_B = key_B.view(batch, self.n_head, self.head_dim, height, width)
        value_B = value_B.view(batch, self.n_head, self.head_dim, height, width)

        # Compute cross-attention for branch A (using B's query)
        out_A = torch.einsum(
            "bnhcw, bnhyx -> bnhwyx", query_B, key_A
        ).contiguous() / math.sqrt(self.head_dim)
        out_A = out_A.view(batch, self.n_head, height, width, -1)
        out_A = torch.softmax(out_A, -1)
        print(out_A.shape)
        out_A = out_A.view(batch, self.n_head, height, width, height, width)
        out_A = torch.einsum("bnhwyx, bnhyx -> bnhcw", out_A, value_A).contiguous()
        out_A = out_A.view(batch, channel, height, width)
        out_A = self.out(out_A) + x_A

        # Compute cross-attention for branch B (using A's query)
        out_B = torch.einsum(
            "bnhcw, bnhyx -> bnhwyx", query_A, key_B
        ).contiguous() / math.sqrt(self.head_dim)
        out_B = out_B.view(batch, self.n_head, height, width, -1)
        out_B = torch.softmax(out_B, -1)
        out_B = out_B.view(batch, self.n_head, height, width, height, width)
        out_B = torch.einsum("bnhwyx, bnhyx -> bnhcw", out_B, value_B).contiguous()
        out_B = out_B.view(batch, channel, height, width)
        out_B = self.out(out_B) + x_B

        return out_A, out_B


#############################################################################
'''
@misc{zamir2022restormerefficienttransformerhighresolution,
      title={Restormer: Efficient Transformer for High-Resolution Image Restoration}, 
      author={Syed Waqas Zamir and Aditya Arora and Salman Khan and Munawar Hayat and Fahad Shahbaz Khan and Ming-Hsuan Yang},
      year={2022},
      eprint={2111.09881},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2111.09881}, 
}
'''
## Layer Norm
def to_3d(x):
    # Convert an image tensor to sequence form: (b, c, h, w) -> (b, h*w, c)
    return rearrange(x, "b c h w -> b (h w) c")

def to_4d(x, h, w):
    # Restore a sequence tensor to image form: (b, h*w, c) -> (b, c, h, w)
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        # Learnable weight
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        # Compute the variance and normalize
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        # Learnable weight and bias
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        # Compute the mean and variance, then normalize
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        # Pick the normalization flavor based on LayerNorm_type
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        # Convert to sequence form, normalize, then restore the shape
        return to_4d(self.body(to_3d(x)), h, w)

## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        # Expand the channel count
        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        # Depthwise convolution
        self.dwconv = nn.Conv2d(
            hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=bias
        )
        # Squeeze the channel count back
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x = F.gelu(x)
        x = self.project_out(x)
        return x

## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        # Generate Q, K, V
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        # Depthwise conv to enhance Q, K, V
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias
        )
        # Output projection
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        # Generate and enhance Q, K, V
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        # Rearrange into multi-head form
        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        # Normalize Q and K
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        # Weighted aggregation of V
        out = attn @ v
        # Restore the image shape
        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )
        # Output projection
        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        # First normalization and attention
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        # Second normalization and feed-forward network
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        # Attention with a residual connection
        x = x + self.attn(self.norm1(x))
        # Feed-forward network with a residual connection
        x = x + self.ffn(self.norm2(x))
        return x
###########################################
class emptyModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
class SpectralWiseAttention(nn.Module):
    def __init__(self, dim, bias=False):
        super(SpectralWiseAttention, self).__init__()
        # Learnable scaling parameter sigma
        self.sigma = nn.Parameter(torch.ones(1, 1))
        # Linear layer generating the query, key, and value vectors (Q, K, V)
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        # Output projection layer
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        # Unpack the input shape: batch (b), channels (c), height (h), width (w)
        b, c, h, w = x.shape

        # Reshape the input to (b, h*w, c), i.e. flatten the spatial dimensions
        x = x.view(b, c, -1).permute(0, 2, 1)  # b, h*w, c
        # Generate the Q, K, V vectors
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # L2-normalize Q and K for stability
        q = torch.nn.functional.normalize(q, dim=1)
        k = torch.nn.functional.normalize(k, dim=1)

        # Compute attention scores: K^T * Q * sigma
        attn = (k.transpose(-2, -1) @ q) * self.sigma
        # Apply softmax to obtain the attention weights
        attn = attn.softmax(dim=-1)
        # Weight V by the attention weights and project through the linear layer
        out = self.linear(v @ attn).permute(0, 2, 1).view(b, c, h, w)

        return out


class SpectralAttentionBlock(nn.Module):
    def __init__(self, dim, bias=False, LayerNorm_type="WithBias"):
        super(SpectralAttentionBlock, self).__init__()
        # Use LayerNorm only when configured
        if LayerNorm_type is None:
            self.norm = emptyModule()
        else:
            self.norm = LayerNorm(dim, LayerNorm_type=LayerNorm_type)
        # 1x1 convs for channel mixing
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        # Spectral attention module
        self.specatt = SpectralWiseAttention(dim, bias)

    def forward(self, x):
        # Residual connection
        res = x
        # Normalize
        x = self.norm(x)
        # 1x1 conv
        x = self.conv1(x)
        # Spectral attention
        x = self.specatt(x)
        # 1x1 conv
        x = self.conv2(x)
        # Residual connection
        x = x + res
        return x


class SERT_ChannelAttention2D(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16, memory_blocks=128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.subnet = nn.Linear(num_feat, num_feat // squeeze_factor)
        self.upnet = nn.Sequential(
            nn.Linear(num_feat // squeeze_factor, num_feat),
            nn.Sigmoid()
        )
        self.mb = nn.Parameter(torch.randn(num_feat // squeeze_factor, memory_blocks))
        self.low_dim = num_feat // squeeze_factor

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        low_rank_f = self.subnet(y).unsqueeze(2)
        mbg = self.mb.unsqueeze(0).repeat(b, 1, 1)
        f1 = torch.bmm(low_rank_f.transpose(1, 2), mbg)
        f_dic_c = F.softmax(f1 * (self.low_dim ** -0.5), dim=-1)
        y1 = torch.bmm(f_dic_c, mbg.transpose(1, 2))
        y2 = self.upnet(y1.squeeze(1))
        return y2.view(b, c, 1, 1)

class LowRankSpectralAttentionBlock(nn.Module):
    def __init__(self, dim, bias=False, LayerNorm_type="WithBias", squeeze_factor=16, memory_blocks=128):
        super().__init__()
        self.norm = LayerNorm(dim, LayerNorm_type=LayerNorm_type) if LayerNorm_type else emptyModule()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.specatt = SpectralWiseAttention(dim, bias=bias)
        self.channel_attn = SERT_ChannelAttention2D(dim, squeeze_factor=squeeze_factor, memory_blocks=memory_blocks)

    def forward(self, x):
        res = x
        x = self.norm(x)
        x = self.conv1(x)
        x = self.specatt(x)
        x = self.conv2(x)
        attn_weights = self.channel_attn(x)
        x = x * attn_weights
        x = x + res
        return x