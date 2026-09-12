import torch
import torch.nn as nn
from .rope_embedding import precompute_rope_angles, apply_rope
import modules.config as config

class CausalSelfAttentionWithKvCache(nn.Module):
    def __init__(self, embed_dim, num_heads, max_seq_len=config.MAX_SEQ_LEN):
        super().__init__()
        self.embed_dim = embed_dim
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.linear_q = nn.Linear(embed_dim, embed_dim)
        self.linear_k = nn.Linear(embed_dim, embed_dim)
        self.linear_v = nn.Linear(embed_dim, embed_dim)
        self.linear_out = nn.Linear(embed_dim, embed_dim)

        self.max_seq_len = max_seq_len
        self.register_buffer(
            "rope_angles",
            precompute_rope_angles(self.max_seq_len, self.head_dim),
            persistent=False,
        )

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, x, kv_cache=None):
        batch_size, seq_length, _ = x.size()

        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        past_len = kv_cache[0].size(2) if kv_cache is not None else 0
        q = apply_rope(q, self.rope_angles, offset=past_len)
        k = apply_rope(k, self.rope_angles, offset=past_len)

        total_seq_length = seq_length + past_len

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv_cache = (k, v)

        attention_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if seq_length > 1:
            mask = torch.full((seq_length, total_seq_length), float('-inf'), device=x.device)
            mask = torch.triu(mask, diagonal = total_seq_length - seq_length + 1)
            attention_weights += mask
        
        attention_weights = torch.softmax(attention_weights, dim=-1)

        output = torch.matmul(attention_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)
        output = self.linear_out(output)

        return output, new_kv_cache