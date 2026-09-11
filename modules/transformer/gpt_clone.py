import torch.nn as nn
from .embedding_layer import EmbeddingLayer
from .transformer_module import TransformerModule

class GPTClone(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, attention_module, mlp_ratio=4):
        super().__init__()
        self.embedding_layer = EmbeddingLayer(vocab_size, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerModule(embed_dim, num_heads, attention_module, mlp_ratio)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)
        self.output_head = nn.Linear(embed_dim, vocab_size, bias=False) # stores weights in out_features, in_features format
        self.output_head.weight = self.embedding_layer.token_embedding.weight

    def forward(self, x, kv_caches=None):
        x = self.embedding_layer(x)
        if kv_caches is None:
            kv_caches = [None] * len(self.blocks)
        
        new_kv_caches = []
        for block, kv_cache in zip(self.blocks, kv_caches):
            x, new_kv_cache = block(x, kv_cache)
            new_kv_caches.append(new_kv_cache)

        x = self.final_norm(x)
        logits = self.output_head(x)
        return logits, new_kv_caches