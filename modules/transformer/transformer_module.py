import torch
import torch.nn as nn

class TransformerModule(nn.Module):
    def __init__(self, embed_dim, num_heads, attention_module, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attention_module = attention_module(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim*mlp_ratio, embed_dim)
        )

    def forward(self, x, kv_cache=None):
        attn_output, new_kv_cache = self.attention_module(self.norm1(x), kv_cache)
        x = x + attn_output
        x = self.mlp(self.norm2(x)) + x
        return x, new_kv_cache
