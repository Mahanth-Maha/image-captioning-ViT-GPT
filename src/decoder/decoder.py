import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from model.transformer import RMSNorm, DecoderBlockTransformer, MultiHeadSelfAttention, MultiHeadCrossAttention, FusedFNNSwiGLU


# class CaptionDecoderBlock(nn.Module):
#     def __init__(self, model_dimension, context_length, n_heads, ffn_hid_dim, n_kv_heads=None, dropout=0.1):
#         super().__init__()
        
#         self.self_attn_norm = RMSNorm(model_dimension)
#         self.self_attn = MultiHeadSelfAttention(model_dimension, context_length, n_heads, n_kv_heads, dropout, is_causal=True)
        
#         self.cross_attn_norm = RMSNorm(model_dimension)
#         self.cross_attn = MultiHeadCrossAttention(model_dimension, n_heads, n_kv_heads, dropout)
        
#         self.ffn_norm = RMSNorm(model_dimension)
#         self.ffn = FusedFNNSwiGLU(model_dimension, ffn_hid_dim, dropout)
        
#     def forward(self, x, vision_features, start_pos=0, self_kv_cache=None, cross_kv_cache=None,causal_mask=None,cross_attn_mask=None):
#         x = x + self.self_attn(self.self_attn_norm(x), start_pos=start_pos, kv_cache=self_kv_cache)
#         x = x + self.cross_attn(self.cross_attn_norm(x), vision_features, mem_kv_cache=cross_kv_cache,attn_mask=cross_attn_mask)
#         x = x + self.ffn(self.ffn_norm(x))    
#         return x


class CaptionDecoder(nn.Module):
    def __init__(self, vocab_size, context_length, model_dimension, n_heads, n_layers, ffn_hid_dim, n_kv_heads=None, dropout=0.1, tie_weights=True, init_std=0.02, use_checkpoint=True, checkpoint_ratio=0.5):
        super().__init__()
        
        self.n_layers = n_layers
        self.use_checkpoint = use_checkpoint
        self.ckpt_start = int(n_layers * checkpoint_ratio)
        self.context_length = context_length
        self.token_emb = nn.Embedding(vocab_size, model_dimension)
        self.blocks = nn.ModuleList([
            # CaptionDecoderBlock(
            DecoderBlockTransformer(
                model_dimension,
                context_length,
                n_heads,
                ffn_hid_dim,
                n_kv_heads,
                dropout
            ) for _ in range(n_layers)
        ])
        
        self.final_norm = RMSNorm(model_dimension)
        self.lm_head = nn.Linear(model_dimension, vocab_size, bias=False)
        
        if tie_weights:
            self.lm_head.weight = self.token_emb.weight
            
        self._init_weights(init_std)
        
    def _init_weights(self, std_val=0.02):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std_val)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std_val)
        scale = ((3 * self.n_layers) ** -0.5)
        for block in self.blocks:
            if hasattr(block.self_attn, 'proj_attn'):
                block.self_attn.proj_attn.weight.data.mul_(scale)
            if hasattr(block.cross_attn, 'proj_attn'):
                block.cross_attn.proj_attn.weight.data.mul_(scale)
            if hasattr(block.ffn, 'proj_ffn'):
                block.ffn.proj_ffn.weight.data.mul_(scale)
    
    def forward(self, input_ids, vision_features, start_pos=0, self_kv_caches=None, cross_kv_caches=None, causal_mask=None, cross_attn_mask=None):
        x = self.token_emb(input_ids)
        # print(f'[Debug] CaptionDecoder | {input_ids.shape = }')
        # print(f'[Debug] CaptionDecoder | {vision_features.shape = }')
        for i, block in enumerate(self.blocks):
            if self.training and self.use_checkpoint and i >= self.ckpt_start:
                def _decode_pass_func(t):
                    return block(
                        t, 
                        vision_features, 
                        start_pos,
                        None if self_kv_caches is None else self_kv_caches[i],
                        None if cross_kv_caches is None else cross_kv_caches[i],
                        causal_mask,
                        cross_attn_mask
                    )
                x = checkpoint(_decode_pass_func, x, use_reentrant=False)
            else:
                x = block(
                    x,
                    vision_features,
                    start_pos,
                    None if self_kv_caches is None else self_kv_caches[i],
                    None if cross_kv_caches is None else cross_kv_caches[i],
                    causal_mask,
                    cross_attn_mask
                )
        
        x = self.final_norm(x)
        logits = self.lm_head(x)
        # print(f'[Debug] Text Decoder: {logits.shape = }')
        return logits
    
    @torch.no_grad()
    def generate(self, vision_features, start_tokens, max_length, temperature=1.0, top_k=None, top_p=None,use_kv_cache=True):
        self.eval()
        device = vision_features.device
        batch_size = vision_features.size(0)
        
        generated = start_tokens.clone()
        
        if use_kv_cache:
            self_kv_caches = [dict() for _ in range(self.n_layers)]
            cross_kv_caches = [dict() for _ in range(self.n_layers)]
        else:
            self_kv_caches = cross_kv_caches = None
            
        for step in range(max_length):
            if use_kv_cache and step > 0:
                input_ids = generated[:, -1:]
                start_pos = step
            else:
                input_ids = generated[:, -self.context_length:]
                start_pos = 0
            
            logits = self.forward(
                input_ids,
                vision_features,
                start_pos=start_pos,
                self_kv_caches=self_kv_caches,
                cross_kv_caches=cross_kv_caches
            )
            
            next_logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                top_k_values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < top_k_values[:, [-1]]] = -float('inf')
            
            if top_p is not None and 0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_mask = cumulative_probs > top_p
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = False
                
                indices_to_remove = sorted_mask.scatter(1, sorted_indices, sorted_mask)
                next_logits = next_logits.masked_fill(indices_to_remove, -float('inf'))
            
            probs = F.softmax(next_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_tokens], dim=-1)
        
        self.train()
        return generated