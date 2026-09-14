import torch
import torch.nn as nn
from modules import config
from modules.attention.rope_embedding import precompute_rope_angles, apply_rope

class GroupedQueryAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_kv_heads=config.NUM_KV_HEADS, max_seq_len=config.MAX_SEQ_LEN):
        super().__init__()

        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.num_kv_heads = num_kv_heads
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        self.max_seq_len = max_seq_len

        self.linear_q = nn.Linear(embed_dim, embed_dim)
        self.linear_k = nn.Linear(embed_dim, self.head_dim * num_kv_heads)
        self.linear_v = nn.Linear(embed_dim, self.head_dim * num_kv_heads)
        self.linear_out = nn.Linear(embed_dim, embed_dim)

        self.register_buffer(
            "rope_angles",
            precompute_rope_angles(self.max_seq_len, self.head_dim),
            persistent=False,
        )

    def split_heads(self, x, batch_size, num_heads):
        x = x.view(batch_size, -1, num_heads, self.head_dim)
        return x.transpose(1, 2)
    
    def forward(self, x, kv_cache=None):
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)

        q = self.split_heads(q, batch_size, self.num_heads)
        k = self.split_heads(k, batch_size, self.num_kv_heads)
        v = self.split_heads(v, batch_size, self.num_kv_heads)

        past_len = kv_cache[0].size(2) if kv_cache is not None else 0
        q = apply_rope(q, self.rope_angles, offset=past_len)
        k = apply_rope(k, self.rope_angles, offset=past_len)

        total_seq_length = seq_len + past_len

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv_cache = (k, v)
        repetitions = self.num_heads // self.num_kv_heads
        k = k.unsqueeze(2).expand(-1, -1, repetitions, -1, -1).reshape(batch_size, self.num_heads, -1, self.head_dim) # expand work only on -1 dimension but is effective
        v = v.unsqueeze(2).expand(-1, -1, repetitions, -1, -1).reshape(batch_size, self.num_heads, -1, self.head_dim)

        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)

        if seq_len > 1:
            mask = torch.full((seq_len, total_seq_length), float('-inf'), device=x.device)
            mask = torch.triu(mask, diagonal=total_seq_length - seq_len + 1)
            attention_scores += mask

        attention_probs = torch.softmax(attention_scores, dim=-1)

        outputs = torch.matmul(attention_probs, v)
        outputs = outputs.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        return self.linear_out(outputs), new_kv_cache