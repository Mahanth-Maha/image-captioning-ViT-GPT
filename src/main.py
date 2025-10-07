#!/usr/bin/python

# Author : 
#    Mahanth Yalla,
#    M.Tech - Artificial Intelligence 
#    Vision \& AI Lab, CDS Dept.,
#    Indian Institute of Science.

from dotenv import load_dotenv
load_dotenv() 


import warnings
# warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


import os, yaml, time, json, math, pytz
import argparse
import platform
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime as dt

import torch
import torch.nn as nn
import torch.nn.functional as F

from rich.box import HEAVY
from rich.panel import Panel
from rich.console import Console
console = Console()

from training import utils as ut
from training.utils import convert2hr2, get_coco_dataloader, get_tokenizer, load_config_from_yaml, merge_configs, overrides_to_dict
from model.vit_based_img_cap_model import ViTImageCaptioningModel
from model.model import ModelFactory


def parse_args():
    parser = argparse.ArgumentParser(description="Model training config overrides")
    parser.add_argument('-c',"--config_file", type=str, default=None, help="Optional additional YAML config file to override defaults")
    parser.add_argument('-z',"--model_config", type=str, default='DEFAULT', help="Optional additional name of YAML config file to override defaults")
    parser.add_argument('-b',"--batch_size", type=int, default=None, help="Config: batch_size")
    parser.add_argument('-v','--model_type', type=str, default=None)
    parser.add_argument('-m',"--max_steps", type=int, default=None, help="Config: max_steps")
    parser.add_argument('-a',"--trainer.accum_steps", type=int, default=None, help="Config: accum_steps ")
    parser.add_argument('-w',"--trainer.train_time_warmup", type=int, default=None, help="Config: train_time_warmup ")
    parser.add_argument('-p',"--trainer.precomputed_train_time", type=float, default=None, help="Config: precomputed ? train_time_warmup ")
    parser.add_argument('-l',"--logging.log_dir", type=str, help="logging dir")
    parser.add_argument('-e',"--trainer.ema_decay", type=float, default=None, help="Config: ema_decay ")
    parser.add_argument('-r',"--trainer.resume_train", action="store_true", default=None, help="Resume ?")
    parser.add_argument('--learning_rate', type=float, default=None)
    return parser.parse_args()


def main():
    console.print(f"[yellow]🧭 Torch: {torch.__version__}, CUDA: {torch.version.cuda}, Python: {platform.python_version()}[/yellow]")
    args = parse_args()
    project_root = os.getenv(f"PROJECT_ROOT")
    default_config = os.getenv(f"CONFIG_{args.model_config}")
    
    base_config = load_config_from_yaml(os.path.join(project_root,default_config))
        
    if args.config_file:
        user_config = load_config_from_yaml(args.config_file)
        base_config = merge_configs(base_config, user_config)
    
    cli_overrides = overrides_to_dict(args)
    final_config = merge_configs(base_config, cli_overrides)
        
    model_config = final_config.get("model", {})
    encoder_config = model_config.get("encoder", {})
    decoder_config = model_config.get("decoder", {})

    decoder_context_length = decoder_config.get("context_length", 32)
    batch_size = final_config.get("batch_size", 1)
    decoder_vocab_size = decoder_config.get("vocab_size", 16384)
    vision_img_resize = encoder_config.get("img_resize", 224)
    
    tokenizer = get_tokenizer(decoder_vocab_size)
    train_loader = get_coco_dataloader(
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_seq_len=decoder_context_length + 1,
        resize_to=vision_img_resize,
        datatype='train'
        )
    val_loader = get_coco_dataloader(
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_seq_len=decoder_context_length + 1,
        resize_to=vision_img_resize,
        datatype='val'
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if final_config["device"] =='cuda' else torch.device("cpu") 


    model_kwargs = {
        'vocab_size': decoder_vocab_size,
        'context_length': decoder_context_length,
        'img_size': vision_img_resize,
        'patch_size': model_config.get('patch_size', 16),
        'vision_embed_dim': encoder_config.get('dim', 768),
        'text_embed_dim': decoder_config.get('dim', 768),
        'vision_n_heads': encoder_config.get('n_heads', 12),
        'vision_n_layers': encoder_config.get('Nx', 12),
        'vision_ffn_dim': encoder_config.get('ffn_dim', 3072),
        'text_n_heads': decoder_config.get('n_heads', 8),
        'text_n_layers': decoder_config.get('Nx', 6),
        'text_ffn_dim': decoder_config.get('ffn_dim', 2304),
        'n_kv_heads': model_config.get('n_kv_heads'),
        'dropout': model_config.get('dropout', 0.1),
        'tie_weights': model_config.get('tie_weights', True),
        'init_std': model_config.get('init_std', 0.02),
        'use_checkpoint': model_config.get('use_checkpoint', True),
        'checkpoint_ratio': model_config.get('checkpoint_ratio', 0.5)
    }
    print(f'{args.model_type=}')
    if args.model_type in ['vit_base_gpt_small', 'vit_large_gpt_medium', 'vit_small_gpt_micro']:
        model = ModelFactory.create_model(args.model_type, **model_kwargs)
        print(f"[Info] Created model using ModelFactory for type: {args.model_type}")
    else:
        model = ViTImageCaptioningModel(**model_kwargs)
        print(f"[Info] Created model using Config and Command line args.")
    
    model = model.to(device)
    
    param_counts = model.get_params_count()
    total_params = param_counts['total']
    
    console.print(Panel.fit(
        f"[bold cyan]📌 Model:[/bold cyan] {args.model_type}\n"
        f"[bold]Vision Encoder:[/bold] {param_counts['vision_encoder']:,} ({convert2hr2(param_counts['vision_encoder'])})\n"
        f"[bold]Text Decoder:[/bold] {param_counts['text_decoder']:,} ({convert2hr2(param_counts['text_decoder'])})\n"
        f"[bold]Vision Projection:[/bold] {param_counts['vision_proj']:,} ({convert2hr2(param_counts['vision_proj'])})\n"
        f"[bold cyan]Total Parameters:[/bold cyan] [bold]{total_params:,}[/bold] [cyan]({convert2hr2(total_params)})[/cyan]\n"
        f"[bold]Context Length:[/bold] {decoder_context_length} | [bold]Batch Size:[/bold] {batch_size}\n"
        f"[bold]Image Size:[/bold] {vision_img_resize}x{vision_img_resize} | [bold]Vocab Size:[/bold] {decoder_vocab_size}\n"
        f"[bold]Device:[/bold] {device}",
        title="🧠 Image Captioning Model Summary",
        border_style="cyan"
    ))

    console.print("\n[yellow]Testing model with sample batch...[/yellow]")
    try:
        sample_batch = next(iter(train_loader))
        images, captions = sample_batch
        images = images.to(device)
        captions = captions.to(device)
        
        # example_image = train_loader.dataset.decode_tensor(images[0])
        # example_captions = tokenizer.decode_tensor(captions[0], skip_special_tokens=True)
        # plt.imshow(example_image)
        # plt.axis('off')
        # plt.title(f"{example_captions}")
        # plt.savefig("example_image.png")
         
        with torch.no_grad():
            output_logits = model(images, captions[:, :-1])
            labels = captions[:, 1:]
            logits_flat = output_logits.reshape(-1, output_logits.size(-1))
            labels_flat = labels.reshape(-1)

            loss = F.cross_entropy(
                logits_flat,
                labels_flat,
                ignore_index=tokenizer.get_token_id(ut.TOKEN_PAD)
            )
        
        console.print(f"[green]✅ Model forward pass successful![/green]")
        console.print(f"[bold]Input shapes:[/bold] Images: {list(images.shape)}, Captions: {list(captions.shape)}")
        console.print(f"[bold]Output logits shape:[/bold] {list(output_logits.shape)}")
        console.print(f"[bold green]Loss:[/bold green] {loss}")

    except Exception as e:
        console.print(f"[red]❌ Model test failed: {e}[/red]")
        return


    console.print("\n[bold yellow]🔁 Overfitting on one batch to sanity check...[/bold yellow]")

    images_batch = images.clone().detach()
    captions_batch = captions.clone().detach()
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    pad_token_id = tokenizer.get_token_id(ut.TOKEN_PAD)
    n_steps = 250
    for step in range(1, n_steps + 1):
        optimizer.zero_grad()
        output_logits = model(images_batch, captions_batch[:, :-1])
        labels = captions_batch[:, 1:]
        logits_flat = output_logits.reshape(-1, output_logits.size(-1))
        labels_flat = labels.reshape(-1)
        loss = F.cross_entropy(
            logits_flat,
            labels_flat,
            # ignore_index=pad_token_id
        )
        loss.backward()
        optimizer.step()

        if step % 20 == 0 or step == 1 or step == n_steps:
            console.print(f"[cyan]Step {step:03d}[/cyan] - Loss: {loss.item():.4f}")

    console.print("[bold green]✅ Overfitting loop completed![/bold green]")
    model.eval()
    with torch.no_grad():
        sample_logits = model(images_batch, captions_batch[:, :-1])
        pred_ids = sample_logits.argmax(dim=-1)
        example_pred = tokenizer.decode_tensor(pred_ids[0], skip_special_tokens=True)
        example_gt = tokenizer.decode_tensor(captions_batch[0], skip_special_tokens=True)
        console.print(f"\n[bold yellow]Ground Truth:[/bold yellow] {example_gt}")
        console.print(f"[bold green]Prediction after overfitting:[/bold green] {example_pred}")

    print()
    console.print('[red]Trainer Not Implemented Yet[/red]')
    print()


if __name__ == "__main__":
    start = dt.now()
    main()
    console.print('[bold green]✅ Done! Script Completed Successfuly[/bold green]')
    print(f'⌚ Script Time: {dt.now() - start}')