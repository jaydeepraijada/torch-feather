import math
import torch
from torchfeather.model.model_args import DeepSeekV3ModelArgs

def precompute_freqs_cis(args: DeepSeekV3ModelArgs) -> torch.Tensor:
    dim = args.qk_rope_head_dim #rope is applied to each head independently
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta  #10_000
    factor = args.rope_factor


    # to compute rope we need to compute the freqs
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype = torch.float32)/dim))

    # Create position indices
    t = torch.arange(seqlen)

    # outer product: [positions] x [frequencies]
    freqs = torch.outer(t, freqs)

    # Convert to complex exponentials: e^(i*freq*pos)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    # freqs_shape: [max_seq_len, D/2]
    return freqs_cis  

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) ->  torch.Tensor:
    # x: [B, S, H, D] => [B, S, H, D/2, 2]
    # freqs_cis: [S, D/2]
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2)) # => [B, S, H, D / 2]
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1)) # => [1, S, 1, D / 2]
    y = torch.view_as_real(x * freqs_cis).flatten(3) 
    return y.to(dtype) 

