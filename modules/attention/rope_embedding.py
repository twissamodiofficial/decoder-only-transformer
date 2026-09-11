import torch
    
def precompute_rope_angles(max_seq_len, head_dim, base=10000):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)

    freq = torch.einsum('i,j->ij', t, inv_freq)
    angles = freq.repeat_interleave(2, dim=-1)
    return angles

def apply_rope(x, angles, offset=0):
    #batch, heads, seq_len, head_dim = x.shape
    seq_len = x.shape[2]
    assert offset + seq_len <= angles.shape[0], (
        f"position range [{offset}, {offset+seq_len}) exceeds precomputed max_seq_len {angles.shape[0]}"
    )
    rope_angles = angles[offset:offset+seq_len, :].to(x.device).unsqueeze(0).unsqueeze(1)  # Shape: (batch, head, seq_len, head_dim)

    cos = rope_angles.cos()
    sin = rope_angles.sin()

    x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2)  # Reshape to (batch_size, num_heads, seq_len, head_dim/2, 2)

    x0 = x_reshaped[..., 0]
    x1 = x_reshaped[..., 1]

    x_partner = torch.stack([-x1, x0], dim=-1)

    x_partner = x_partner.flatten(-2)

    x_rotated = (x.float() * cos + x_partner * sin).type_as(x)
    return x_rotated
