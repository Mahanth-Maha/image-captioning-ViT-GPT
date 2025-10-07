from model.vit_based_img_cap_model import ViTImageCaptioningModel

class ModelFactory:    
    @staticmethod
    def create_model(model_type, **kwargs):
        configs = {
            'vit_small_gpt_micro': dict(
                vision_embed_dim = 128,
                text_embed_dim = 128,
                vision_n_heads = 8,
                vision_n_layers = 6,
                vision_ffn_dim = 384,
                text_n_heads = 8,
                text_n_layers = 6,
                text_ffn_dim = 384,
                img_size = 224,
                patch_size = 16,
            ),
            'vit_base_gpt_small': dict(
                vision_embed_dim = 768,
                text_embed_dim = 768,
                vision_n_heads = 12,
                vision_n_layers = 12,
                vision_ffn_dim = 3072,
                text_n_heads = 8,
                text_n_layers = 6,
                text_ffn_dim = 2304,
                img_size = 224,
                patch_size = 16,
            ),
            'vit_large_gpt_medium': dict(
                vision_embed_dim = 1024,
                text_embed_dim = 1024,
                vision_n_heads = 16,
                vision_n_layers = 24,
                vision_ffn_dim = 4096,
                text_n_heads = 16,
                text_n_layers = 12,
                text_ffn_dim = 4096,
                img_size = 224,
                patch_size = 16,
            )
        }
        
        if model_type not in configs:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(configs.keys())}")
        
        config = configs[model_type]
        kwargs.update(config)
        if 'vit' in model_type:
            return ViTImageCaptioningModel(**kwargs)
        else:
            raise ValueError(f"Model type {model_type} is not supported in the factory.")
        