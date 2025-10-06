import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

DROPOUT_DEFAULT = 0.1

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len, base = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self._precompute_cos_sin(max_seq_len)
    
    def _precompute_cos_sin(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq) 
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)
    
    def rotate_half(self, x) :
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, q, k, start_pos = 0):
        T = q.shape[1]
        
        if start_pos + T > self.max_seq_len:
            self._precompute_cos_sin(start_pos + T)
            self.max_seq_len = start_pos + T
        
        cos = self.cos_cached[start_pos:start_pos + T]
        sin = self.sin_cached[start_pos:start_pos + T]
        
        cos = cos.to(dtype=q.dtype, device=q.device)[None, :, None, :]
        sin = sin.to(dtype=q.dtype, device=q.device)[None, :, None, :]
        
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        
        return q_embed, k_embed

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device = kwargs.get('device', None) or (args[0] if args else None)
        if device is not None:
            self.cos_cached = self.cos_cached.to(device)
            self.sin_cached = self.sin_cached.to(device)
            self.inv_freq = self.inv_freq.to(device)
        return self

    def ntk_rescale(self, new_max_seq_len):
        # NTK-aware RoPE scaling for longer context lengths. https://arxiv.org/abs/2306.15595
        scale = new_max_seq_len / self.max_seq_len
        self.inv_freq /= scale
        self._precompute_cos_sin(new_max_seq_len)
        self.max_seq_len = new_max_seq_len
        print(f"[RoPE] NTK scaling applied for {new_max_seq_len} context (factor={scale:.2f})")


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, model_dimension, context_length, n_heads, n_kv_heads=None, dropout=DROPOUT_DEFAULT, is_causal=True):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.each_head_size = model_dimension // n_heads
        assert model_dimension % n_heads == 0, "model_dimension must be divisible by n_heads"
        assert self.each_head_size % 2 == 0, "each_head_size must be even for RoPE"
        assert 0 < self.n_kv_heads <= self.n_heads, "n_kv_heads must be > 0 and <= n_heads"
        assert n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.context_length = context_length
        self.dropout = dropout
        self.is_causal = is_causal

        total_proj_dim = (n_heads + 2 * self.n_kv_heads) * self.each_head_size
        self.replication_factor = n_heads // self.n_kv_heads
        self.q_end = self.n_heads * self.each_head_size
        self.k_end = self.q_end + self.n_kv_heads * self.each_head_size

        self.W_qkv = nn.Linear(model_dimension, total_proj_dim, bias=False)
        self.proj_attn = nn.Linear(model_dimension, model_dimension, bias=False)
        self.attn_proj_dropout = nn.Dropout(self.dropout)

        self.rope = RotaryPositionalEmbedding(
            dim=self.each_head_size,
            max_seq_len=context_length,
        )

    def forward(self, x, start_pos=0, kv_cache=None, attn_mask = None):
        B, T, C = x.shape

        QKV = self.W_qkv(x)

        Q = QKV[:, :, :self.q_end]
        K = QKV[:, :, self.q_end: self.k_end]
        V = QKV[:, :, self.k_end:]

        Q = Q.view(B, T, self.n_heads, self.each_head_size)
        K = K.view(B, T, self.n_kv_heads, self.each_head_size)
        V = V.view(B, T, self.n_kv_heads, self.each_head_size)

        Q, K = self.rope(Q, K, start_pos)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        if kv_cache is not None:
            if "k" in kv_cache:
                K = torch.cat([kv_cache["k"], K], dim=2)
                V = torch.cat([kv_cache["v"], V], dim=2)

            if K.size(2) > self.context_length:
                K = K[:, :, -self.context_length:, :]
                V = V[:, :, -self.context_length:, :]

            kv_cache["k"], kv_cache["v"] = K, V

        if self.n_kv_heads < self.n_heads:
            K = K.repeat_interleave(self.replication_factor, dim=1)
            V = V.repeat_interleave(self.replication_factor, dim=1)

        sdpa_mask = None
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                sdpa_mask = torch.zeros_like(attn_mask, dtype=Q.dtype).masked_fill(attn_mask, float("-inf"))
            else:
                sdpa_mask = attn_mask

        sdpa = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=sdpa_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.is_causal
        )

        output = sdpa.transpose(1, 2).contiguous().view(B, T, C)
        return self.attn_proj_dropout(self.proj_attn(output))

class FusedFNNSwiGLU(nn.Module):
    def __init__(self, model_dimension, ffn_hid_dim, dropout=DROPOUT_DEFAULT):
        super().__init__()
        self.proj_silu = nn.Linear(model_dimension, 2 * ffn_hid_dim, bias=False)    
        self.proj_ffn = nn.Linear(ffn_hid_dim, model_dimension, bias=False)    
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        a, b = self.proj_silu(x).chunk(2, dim=-1)
        x = F.silu(a) * b
        return self.dropout_layer(self.proj_ffn(x))

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        orig_dtype = x.dtype
        x_float = x.to(torch.float32)
        var = x_float.pow(2).mean(-1, keepdim=True)
        x_norm = x_float * torch.rsqrt(var + self.eps)
        return (self.weight * x_norm).to(orig_dtype)


class DecoderBlock(nn.Module):
    def __init__(self, model_dimension, context_length, n_heads, ffn_hid_dim, n_kv_heads=None, dropout=DROPOUT_DEFAULT):
        super().__init__()
        self.attn_norm = RMSNorm(model_dimension)
        self.attn = MultiHeadSelfAttention(model_dimension, context_length, n_heads, n_kv_heads, dropout)
        self.ffn_norm = RMSNorm(model_dimension)
        self.ffn = FusedFNNSwiGLU(model_dimension, ffn_hid_dim, dropout)

    def forward(self, x, start_pos = 0, kv_cache=None):
        x = x + self.attn(self.attn_norm(x), start_pos, kv_cache=kv_cache)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, context_length, model_dimension, n_heads, Nx_blocks, ffn_hid_dim, n_kv_heads =None, dropout=DROPOUT_DEFAULT, tie_weights=True, init_std_val = 0.02, use_checkpoint=True, checkpoint_ratio = 0.5):
        super().__init__()
        self.Nx_blocks = Nx_blocks
        self.use_checkpoint = use_checkpoint
        self.ckpt_start = int(Nx_blocks * checkpoint_ratio)
        self.context_length = context_length
        self.token_emb = nn.Embedding(vocab_size, model_dimension)
        self.blocks = nn.ModuleList([
            DecoderBlock(
                model_dimension, 
                context_length,
                n_heads, 
                ffn_hid_dim, 
                n_kv_heads, 
                dropout
            ) for _ in range(Nx_blocks)
        ])
        self.final_norm = RMSNorm(model_dimension)
        self.lm_head = nn.Linear(model_dimension, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.token_emb.weight
        self._init_weights(init_std_val)

    def _init_weights(self, std_val = 0.02):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std_val)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std_val)
        scale =  ((2 * self.Nx_blocks) ** -0.5)
        for b in self.blocks:
            b.attn.proj_attn.weight.data.mul_(scale)
            b.ffn.proj_ffn.weight.data.mul_(scale)
            
    def forward(self, input_ids, start_pos=0):
        x = self.token_emb(input_ids)
        for i, block in enumerate(self.blocks):
            if self.training and self.use_checkpoint and i >= self.ckpt_start:
                x = checkpoint(lambda t: block(t, start_pos), x, use_reentrant=False)
            else:
                x = block(x, start_pos)
        x = self.final_norm(x)
        return self.lm_head(x)
    
    
    def _forward_with_cache(self, x, start_pos, kv_caches):
        x = self.token_emb(x)
        for i, block in enumerate(self.blocks):
            x = block(x, start_pos=start_pos, kv_cache=kv_caches[i])
        x = self.final_norm(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, x, max_pred_tokens, temp=1.0, top_k=None, top_p=None, kv_cache = True):
        self.eval()
        target_device = x.device
        kv_caches = [dict() for _ in range(self.Nx_blocks)] if kv_cache else None
        
        gen = torch.Generator(device=target_device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for token_iter in range(max_pred_tokens):
                if kv_cache:
                    logits = self._forward_with_cache(x[:, -1:], token_iter, kv_caches)
                else:
                    logits = self(x[:, -self.context_length:])
                logits = logits[:, -1, :] / temp

                if top_k is not None:
                    top_k_vals, top_k_idxs = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < top_k_vals[:, [-1]]] = -float('Inf')

                if top_p is not None and 0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    sorted_mask = cumulative_probs > top_p

                    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                    sorted_mask[..., 0] = 0

                    indices_to_remove = sorted_mask.scatter(1, sorted_indices, sorted_mask)
                    logits = logits.masked_fill(indices_to_remove, -float('Inf'))

                prob_dist = F.softmax(logits, -1)
                x = torch.cat([x, torch.multinomial(prob_dist, 1, generator=gen)], -1).to(target_device)
            self.train()
            return x


class EncoderBlock(nn.Module):
    def __init__(self, model_dimension, context_length, n_heads, ffn_hid_dim, n_kv_heads=None, dropout=DROPOUT_DEFAULT):
        super().__init__()
        self.attn_norm = RMSNorm(model_dimension)
        self.attn = MultiHeadSelfAttention(model_dimension, context_length, n_heads, n_kv_heads, dropout, is_causal=False)
        self.ffn_norm = RMSNorm(model_dimension)
        self.ffn = FusedFNNSwiGLU(model_dimension, ffn_hid_dim, dropout)

    def forward(self, x, attn_mask= None):
        x = x + self.attn(self.attn_norm(x), start_pos=0, kv_cache=None, attn_mask=attn_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class EncoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, context_length, model_dimension, n_heads, Nx_blocks, ffn_hid_dim, n_kv_heads=None, dropout=DROPOUT_DEFAULT, tie_weights=False, init_std_val=0.02):
        super().__init__()
        self.Nx_blocks = Nx_blocks
        self.context_length = context_length

        self.token_emb = nn.Embedding(vocab_size, model_dimension)
        self.blocks = nn.ModuleList([
            EncoderBlock(
                model_dimension,
                context_length,
                n_heads,
                ffn_hid_dim,
                n_kv_heads,
                dropout
            ) for _ in range(Nx_blocks)
        ])
        self.final_norm = RMSNorm(model_dimension)

        self.output_proj = None
        if tie_weights:
            self.output_proj = nn.Linear(model_dimension, vocab_size, bias=False)
            self.output_proj.weight = self.token_emb.weight

        self._init_weights(init_std_val)

    def _init_weights(self, std_val=0.02):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std_val)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std_val)

        scale = ((2 * self.Nx_blocks) ** -0.5)
        for b in self.blocks:
            b.attn.proj_attn.weight.data.mul_(scale)
            b.ffn.proj_ffn.weight.data.mul_(scale)

    def forward(self, input_ids, attn_mask = None):
        x = self.token_emb(input_ids)
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)
        x = self.final_norm(x)
        if self.output_proj is not None:
            return self.output_proj(x)
        return x
    

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, model_dimension, n_heads, n_kv_heads=None, dropout=DROPOUT_DEFAULT):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.each_head_size = model_dimension // n_heads
        assert model_dimension % n_heads == 0, "model_dimension must be divisible by n_heads"
        assert 0 < self.n_kv_heads <= self.n_heads, "n_kv_heads must be > 0 and <= n_heads"
        assert n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.dropout = dropout
        self.replication_factor = n_heads // self.n_kv_heads

        self.W_q = nn.Linear(model_dimension, n_heads * self.each_head_size, bias=False)
        self.W_kv = nn.Linear(model_dimension, 2 * self.n_kv_heads * self.each_head_size, bias=False)
        self.proj_attn = nn.Linear(model_dimension, model_dimension, bias=False)
        self.attn_proj_dropout = nn.Dropout(self.dropout)

    def forward(self, x_q, x_mem, mem_kv_cache=None, attn_mask = None):
        B, Tq, C = x_q.shape
        S = x_mem.size(1)

        Q = self.W_q(x_q).view(B, Tq, self.n_heads, self.each_head_size).transpose(1, 2)

        if mem_kv_cache is not None and "k" in mem_kv_cache and "v" in mem_kv_cache:
            K = mem_kv_cache["k"]
            V = mem_kv_cache["v"]
        else:
            KV = self.W_kv(x_mem)
            k_end = self.n_kv_heads * self.each_head_size
            K = KV[:, :, :k_end].view(B, S, self.n_kv_heads, self.each_head_size).transpose(1, 2)
            V = KV[:, :, k_end:].view(B, S, self.n_kv_heads, self.each_head_size).transpose(1, 2)
            if mem_kv_cache is not None:
                mem_kv_cache["k"], mem_kv_cache["v"] = K, V

        if self.n_kv_heads < self.n_heads:
            K = K.repeat_interleave(self.replication_factor, dim=1)
            V = V.repeat_interleave(self.replication_factor, dim=1)

        sdpa_mask = None
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                sdpa_mask = torch.zeros_like(attn_mask, dtype=Q.dtype).masked_fill(attn_mask, float("-inf"))
            else:
                sdpa_mask = attn_mask

        sdpa = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=sdpa_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False
        )
        out = sdpa.transpose(1, 2).contiguous().view(B, Tq, C)
        return self.attn_proj_dropout(self.proj_attn(out))


class DecoderBlockTransformer(nn.Module):
    def __init__(self, model_dimension, context_length, n_heads, ffn_hid_dim, n_kv_heads=None, dropout=DROPOUT_DEFAULT):
        super().__init__()
        self.self_attn_norm = RMSNorm(model_dimension)
        self.self_attn = MultiHeadSelfAttention(model_dimension, context_length, n_heads, n_kv_heads, dropout, is_causal=True)
        
        self.cross_attn_norm = RMSNorm(model_dimension)
        self.cross_attn = MultiHeadCrossAttention(model_dimension, n_heads, n_kv_heads, dropout)
        
        self.ffn_norm = RMSNorm(model_dimension)
        self.ffn = FusedFNNSwiGLU(model_dimension, ffn_hid_dim, dropout)

    def forward(self, x, enc_out, start_pos=0, self_kv_cache=None, mem_kv_cache=None, self_attn_mask = None, cross_attn_mask= None):
        # Self-attn
        x = x + self.self_attn(self.self_attn_norm(x), start_pos=start_pos, kv_cache=self_kv_cache, attn_mask=self_attn_mask)
        # Cross-attn
        x = x + self.cross_attn(self.cross_attn_norm(x), enc_out, mem_kv_cache, attn_mask=cross_attn_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class Transformer(nn.Module):
    def __init__( self, src_vocab_size, tgt_vocab_size, context_length_src, context_length_tgt, model_dimension, n_heads, Nx_encoder, Nx_decoder, ffn_hid_dim, n_kv_heads=None, dropout=DROPOUT_DEFAULT, tie_weights=True, init_std_val=0.02, use_checkpoint=True, checkpoint_ratio=0.5):
        super().__init__()
        self.Nx_encoder = Nx_encoder
        self.Nx_decoder = Nx_decoder
        self.use_checkpoint = use_checkpoint
        self.ckpt_start_enc = int(Nx_encoder * checkpoint_ratio)
        self.ckpt_start_dec = int(Nx_decoder * checkpoint_ratio)
        self.context_length_src = context_length_src
        self.context_length_tgt = context_length_tgt

        self.src_token_emb = nn.Embedding(src_vocab_size, model_dimension)
        self.tgt_token_emb = nn.Embedding(tgt_vocab_size, model_dimension)

        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(
                model_dimension,
                context_length_src,
                n_heads,
                ffn_hid_dim,
                n_kv_heads,
                dropout
            ) for _ in range(Nx_encoder)
        ])
        self.encoder_final_norm = RMSNorm(model_dimension)

        self.decoder_blocks = nn.ModuleList([
            DecoderBlockTransformer(
                model_dimension,
                context_length_tgt,
                n_heads,
                ffn_hid_dim,
                n_kv_heads,
                dropout
            ) for _ in range(Nx_decoder)
        ])
        self.decoder_final_norm = RMSNorm(model_dimension)

        self.lm_head = nn.Linear(model_dimension, tgt_vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.tgt_token_emb.weight

        self._init_weights(init_std_val)

    def _init_weights(self, std_val=0.02):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std_val)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std_val)

        scale_enc = ((2 * self.Nx_encoder) ** -0.5) if self.Nx_encoder > 0 else 1.0
        for b in self.encoder_blocks:
            b.attn.proj_attn.weight.data.mul_(scale_enc)
            b.ffn.proj_ffn.weight.data.mul_(scale_enc)

        scale_dec = ((3 * self.Nx_decoder) ** -0.5) if self.Nx_decoder > 0 else 1.0
        for b in self.decoder_blocks:
            b.self_attn.proj_attn.weight.data.mul_(scale_dec)
            b.cross_attn.proj_attn.weight.data.mul_(scale_dec)
            b.ffn.proj_ffn.weight.data.mul_(scale_dec)

    def encode(self, src_ids, src_attn_mask= None):
        x = self.src_token_emb(src_ids)
        for i, block in enumerate(self.encoder_blocks):
            if self.training and self.use_checkpoint and i >= self.ckpt_start_enc:
                x = checkpoint(lambda t: block(t, attn_mask=src_attn_mask), x, use_reentrant=False)
            else:
                x = block(x, attn_mask=src_attn_mask)
        return self.encoder_final_norm(x)

    def decode(self, tgt_ids, enc_out, start_pos=0,self_kv_caches=None, mem_kv_caches=None,self_attn_mask= None, cross_attn_mask= None):
        x = self.tgt_token_emb(tgt_ids)
        for i, block in enumerate(self.decoder_blocks):
            if self.training and self.use_checkpoint and i >= self.ckpt_start_dec:
                def _decode_pass_func(t):
                    return block(
                        t, enc_out, start_pos,
                        None if self_kv_caches is None else self_kv_caches[i],
                        None if mem_kv_caches is None else mem_kv_caches[i],
                        self_attn_mask, cross_attn_mask
                    )
                x = checkpoint(_decode_pass_func, x, use_reentrant=False)
            else:
                x = block(
                    x, enc_out, start_pos,
                    None if self_kv_caches is None else self_kv_caches[i],
                    None if mem_kv_caches is None else mem_kv_caches[i],
                    self_attn_mask, cross_attn_mask
                )
        x = self.decoder_final_norm(x)
        return self.lm_head(x)

    def forward(self, src_ids, tgt_ids, src_attn_mask= None, tgt_self_attn_mask= None, cross_attn_mask= None):
        enc_out = self.encode(src_ids, src_attn_mask)
        logits = self.decode(tgt_ids, enc_out, start_pos=0,self_kv_caches=None, mem_kv_caches=None, self_attn_mask=tgt_self_attn_mask, cross_attn_mask=cross_attn_mask)
        return logits

