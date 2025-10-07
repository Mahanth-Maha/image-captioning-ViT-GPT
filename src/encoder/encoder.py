import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from model.transformer import RMSNorm, FusedFNNSwiGLU


class ViTMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, n_kv_heads=None, dropout=0.1):
        super().__init__()
        
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.head_dim = embed_dim // n_heads
        
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        assert 0 < self.n_kv_heads <= self.n_heads, "n_kv_heads must be > 0 and <= n_heads"
        assert n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        
        self.replication_factor = n_heads // self.n_kv_heads
        self.dropout = dropout
        
        total_proj_dim = (n_heads + 2 * self.n_kv_heads) * self.head_dim
        self.q_end = self.n_heads * self.head_dim
        self.k_end = self.q_end + self.n_kv_heads * self.head_dim
        
        self.qkv_proj = nn.Linear(embed_dim, total_proj_dim, bias=False)
        self.proj_attn = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        
        qkv = self.qkv_proj(x)
        
        q = qkv[:, :, :self.q_end]
        k = qkv[:, :, self.q_end:self.k_end]
        v = qkv[:, :, self.k_end:]
        
        q = q.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        if self.n_kv_heads < self.n_heads:
            k = k.repeat_interleave(self.replication_factor, dim=1)
            v = v.repeat_interleave(self.replication_factor, dim=1)
        
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False
        )
        
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, C)
        output = self.proj_attn(attn_out)
        
        return self.dropout_layer(output)


class ViTEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_patches, n_heads, ffn_dim, n_kv_heads=None, dropout=0.1):
        super().__init__()
        
        self.attn_norm = RMSNorm(embed_dim)
        self.attn = ViTMultiHeadAttention(embed_dim, n_heads, n_kv_heads, dropout)
        
        self.ffn_norm = RMSNorm(embed_dim)
        self.ffn = FusedFNNSwiGLU(embed_dim, ffn_dim, dropout)
        
    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
    def forward(self, x):
        # (B, C, I, I) -> (B, E, G, G)
        x = self.patch_embed(x)
        # (B, E, G, G) -> (B, E, N_patch)
        x = x.flatten(2)
        # (B, E, N_patch) -> (B, N_patch, E)
        x = x.transpose(1, 2)
        return x

class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        
    def forward(self, x):
        return x + self.pos_embed


class VisionTransformerEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3,embed_dim=768,n_heads=12,n_layers=12,ffn_dim=3072,n_kv_heads=None,dropout=0.1,use_checkpoint=True,checkpoint_ratio=0.5,init_std=0.02):
        super().__init__()
        
        self.n_layers = n_layers
        self.use_checkpoint = use_checkpoint
        self.ckpt_start = int(n_layers * checkpoint_ratio)
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.pos_embed = LearnedPositionalEmbedding(num_patches, embed_dim)
        
        self.blocks = nn.ModuleList([
            ViTEncoderBlock(
                embed_dim,
                num_patches,
                n_heads,
                ffn_dim,
                n_kv_heads,
                dropout
            ) for _ in range(n_layers)
        ])
        
        self.norm = RMSNorm(embed_dim)
        
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        self._init_weights(init_std)
        
    def _init_weights(self, std_val=0.02):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std_val)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        scale = ((2 * self.n_layers) ** -0.5)
        for block in self.blocks:
            block.attn.proj_attn.weight.data.mul_(scale)
            block.ffn.proj_ffn.weight.data.mul_(scale)
    
    def forward(self, x):
        B = x.shape[0]
        # print(f'[Debug] VisionTransformerEncoder | in {x.shape = }')
        x = self.patch_embed(x)
        x = self.pos_embed(x)
        for i, block in enumerate(self.blocks):
            if self.training and self.use_checkpoint and i >= self.ckpt_start:
                x = checkpoint(lambda t: block(t), x, use_reentrant=False)
            else:
                x = block(x)
        
        x = self.norm(x)
        x = self.output_proj(x)
        # print(f'[Debug] VisionTransformerEncoder | out {x.shape = }')
        return x