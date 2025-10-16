import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder.encoder import VisionTransformerEncoder
from decoder.decoder import CaptionDecoder

import constants as cnst

class ViTImageCaptioningModel(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        
        vision_embed_dim=768,
        text_embed_dim=768,
        
        vision_n_heads=6,
        vision_n_layers=12,
        vision_ffn_dim=3072,
        
        vocab_size=16384,
        context_length=128,
        text_n_heads=8,
        text_n_layers=6,
        text_ffn_dim=2304,
        
        n_kv_heads=None,
        dropout=0.1,
        tie_weights=True,
        init_std=0.02,
        use_checkpoint=True,
        checkpoint_ratio=0.5
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.vision_embed_dim = vision_embed_dim
        self.text_embed_dim = text_embed_dim
        
        self.vision_encoder = VisionTransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=vision_embed_dim,
            n_heads=vision_n_heads,
            n_layers=vision_n_layers,
            ffn_dim=vision_ffn_dim,
            n_kv_heads=n_kv_heads,
            dropout=dropout,
            use_checkpoint=use_checkpoint,
            checkpoint_ratio=checkpoint_ratio,
            init_std=init_std
        )
        
        if vision_embed_dim != text_embed_dim:
            self.vision_proj = nn.Linear(vision_embed_dim, text_embed_dim)
        else:
            self.vision_proj = nn.Identity()
        
        self.text_decoder = CaptionDecoder(
            vocab_size=vocab_size,
            context_length=context_length,
            model_dimension=text_embed_dim,
            n_heads=text_n_heads,
            n_layers=text_n_layers,
            ffn_hid_dim=text_ffn_dim,
            n_kv_heads=n_kv_heads,
            dropout=dropout,
            tie_weights=tie_weights,
            init_std=init_std,
            use_checkpoint=use_checkpoint,
            checkpoint_ratio=checkpoint_ratio
        )
        
        if hasattr(self.vision_proj, 'weight'):
            nn.init.normal_(self.vision_proj.weight, mean=0.0, std=init_std)
            if self.vision_proj.bias is not None:
                nn.init.zeros_(self.vision_proj.bias)
    
    def encode_image(self, images):
        vision_features = self.vision_encoder(images)
        vision_features = self.vision_proj(vision_features)
        return vision_features
    
    def forward(self, images, input_ids):
        # print(f'[Debug] Model | {images.shape = }')
        # print(f'[Debug] Model | {input_ids.shape = }')
        vision_features = self.encode_image(images)
        # print(f'[Debug] Model | {vision_features.shape = }')
        logits = self.text_decoder(input_ids=input_ids,vision_features=vision_features)
        # print(f'[Debug] Model | {logits.shape = }')
        return logits
    
    @torch.no_grad()
    def get_params_count(self):
        vision_params = sum(p.numel() for p in self.vision_encoder.parameters())
        text_params = sum(p.numel() for p in self.text_decoder.parameters())
        proj_params = sum(p.numel() for p in self.vision_proj.parameters()) if hasattr(self.vision_proj, 'parameters') else 0
        
        return {
            'vision_encoder': vision_params,
            'text_decoder': text_params,
            'vision_proj': proj_params,
            'total': vision_params + text_params + proj_params
        }

    
    @torch.no_grad()
    def generate_caption(self, images, tokenizer,max_length=50, temperature=1.0, top_k=None, top_p=None,use_kv_cache=True):
        self.eval()
        device = images.device
        batch_size = images.size(0)
        vision_features = self.encode_image(images)
        start_token = tokenizer.special_tokens.get(cnst.TOKEN_SOS, 0)
        start_tokens = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)
        generated = self.text_decoder.generate(
            vision_features=vision_features,
            start_tokens=start_tokens,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            use_kv_cache=use_kv_cache
        )
        
        self.train()
        return generated