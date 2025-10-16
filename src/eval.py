#!/usr/bin/python

# Author : 
#    Mahanth Yalla,
#    M.Tech - Artificial Intelligence 
#    Vision \& AI Lab, CDS Dept.,
#    Indian Institute of Science.

from pathlib import Path
from dotenv import load_dotenv
load_dotenv() 


import warnings
# warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


import os, yaml, time, json, math, pytz, shutil
import argparse
import platform
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime as dt
import datetime
import copy 
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


from rich.box import HEAVY
from rich.panel import Panel
from rich.console import Console
console = Console()

import constants as cnst
import utils as main_ut
from training.utils import convert2hr2, fmt_dt_ist, fmt_time ,create_optimizer, create_scheduler,\
                            get_fused_cross_entropy, load_config_from_yaml, merge_configs, overrides_to_dict
from model.vit_based_img_cap_model import ViTImageCaptioningModel
from model.model import ModelFactory
from training.trainer import ImageCaptioningTrainer, CaptioningTrainerConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Model training config overrides")
    parser.add_argument('-z',"--model_base_config", type=str, default='DEFAULT', help="Optional name of YAML config file to use it as default config")
    parser.add_argument('-c',"--config_file", type=str, default=None, help="Optional additional YAML config file to override defaults")
    parser.add_argument('-mt','--model_type', type=str, default=None, help="Model Types\n1. vit_base_gpt_small\n2. vit_large_gpt_medium\n3. vit_small_gpt_micro")
    
    parser.add_argument('--overfit', action='store_true', default=False, help='Overfit on one batch for debugging')
    parser.add_argument('--train', action='store_true', default=False, help='Train the model')
    parser.add_argument('--test', action='store_true', default=False, help='Test the model on test set')
    parser.add_argument('--overfit-steps', type=int, default=1, help='Overfit on one batch for debugging')
    
    parser.add_argument('-rn','--training.run_name', type=str, default=None, help='Experiment run name')
    parser.add_argument('-l',"--training.log_dir", type=str, default='checkpoints/vit', help="logging dir")
    parser.add_argument('--training.use_tensorboard', action='store_true', default=None, help='Enable TensorBoard logging')
    parser.add_argument('--training.use_wandb', action='store_true', default=None, help='Enable Weights & Biases logging')
    parser.add_argument('--training.wandb_mode', type=str, default='disabled', choices=['online', 'offline', 'disabled'], help='W&B logging mode')
    
    parser.add_argument('-w',"--training.train_time_warmup", type=int, default=None, help="Config: train_time_warmup ")
    parser.add_argument('-p',"--training.precomputed_train_time", type=float, default=None, help="Config: precomputed ? train_time_warmup ")
    
    parser.add_argument('-b',"--batch_size", type=int, default=None, help="Config: batch_size")
    parser.add_argument('-a',"--training.accum_steps", type=int, default=None, help="Config: accum_steps ")
    parser.add_argument('-lr','--training.learning_rate', type=float, default=None, help='Learning rate (overrides config)')
    parser.add_argument('-ms',"--training.max_steps", type=int, default=None, help="Config: max steps")
    parser.add_argument('-me',"--training.epochs", type=int, default=None, help="Config: max epochs")
    parser.add_argument('-ue',"--training.use_epochs",  action='store_true', default=None, help="Config: use epochs")
    
    parser.add_argument('--resume', action='store_true', default=None, help='Auto-resume from latest checkpoint')
    parser.add_argument('--resume_from', type=str, default=None, help='Specific checkpoint path to resume from')
    
    parser.add_argument('--val_every_steps', type=int, default=None, help='Validation frequency in steps')
    parser.add_argument('--save_every_steps', type=int, default=None, help='Checkpoint saving frequency in steps')
    
    parser.add_argument("--gpus", default="", help="Comma-separated GPU ids to use, e.g., 0,1,3")
    parser.add_argument('--parallel_type', type=str, default='DDP', choices=['DP','DDP', 'None'] ,help='DP, DDP (default), None')
    
    return parser.parse_args()


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return True, local_rank
    return False, 0

def is_main_process(is_distributed):
    return (not is_distributed) or (dist.get_rank() == 0)

def main():
    console.print(f"[yellow]🧭 Torch: {torch.__version__}, CUDA: {torch.version.cuda}, Python: {platform.python_version()}[/yellow]")
    is_distributed, local_rank = setup_distributed()
    rank = dist.get_rank() if is_distributed else 0
    is_main_process = (rank == 0)
    if is_main_process:
        time.sleep(2)
    args = parse_args()
    project_root = os.getenv(f"PROJECT_ROOT", ".")
    default_config = os.getenv(f"CONFIG_{args.model_base_config}")
    
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
    
    tokenizer = main_ut.get_tokenizer(
        decoder_vocab_size,
        model = final_config['tokenizer'],
        )
    train_loader = main_ut.get_coco_dataloader(
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_seq_len=decoder_context_length + 1,
        resize_to=vision_img_resize,
        datatype='train'
        )
    
    val_loader = main_ut.get_coco_dataloader(
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_seq_len=decoder_context_length + 1,
        resize_to=vision_img_resize,
        datatype='val'
        )

    test_loader = main_ut.get_coco_dataloader(
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_seq_len=decoder_context_length + 1,
        resize_to=vision_img_resize,
        datatype='test'
        )
    
    console.print(f"[green]✅ Data loaders ready - Train: {len(train_loader)} batches, Val: {len(val_loader)} batches[/green]")
    
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        print(f"💻 Using GPUs: {args.gpus}")
    else:
        print("💻 Using all available GPUs.")
        
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if final_config["device"] =='cuda' else torch.device("cpu") 
    console.print(f"[cyan]🖥️  Using device: {device}[/cyan]")
    
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
    
    if args.model_type in ['vit_base_gpt_small', 'vit_large_gpt_medium', 'vit_small_gpt_micro']:
        model = ModelFactory.create_model(args.model_type, **model_kwargs)
        if is_main_process:
            print(f"\n👉 Created model using ModelFactory for type: {args.model_type} !\n")
    else:
        model = ViTImageCaptioningModel(**model_kwargs)
        args.model_type = model_config['name']
        if is_main_process:
            print(f"\n👉 Created model using given Config overridden with Command line args...\n")
    
    param_counts = model.get_params_count()
    total_params = param_counts['total']
    if is_main_process:
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

        console.print("\n[yellow]🖋️  Testing model with sample batch...(on CPU: please wait for a minute) [/yellow]")
    try:
        sample_batch = next(iter(train_loader))
        images, captions = sample_batch
        
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
                ignore_index=tokenizer.get_token_id(cnst.TOKEN_PAD)
            )
        if is_main_process:
            console.print(f"[green bold]✅ Model forward pass successful![/green bold]")
            console.print(f"[bold]Input shapes:[/bold] \n\tImages: {list(images.shape)}, \n\tCaptions: {list(captions.shape)}")
            console.print(f"[bold]Output logits shape:[/bold] {list(output_logits.shape)}")
            console.print(f"[bold green]Loss:[/bold green] {loss}")

    except Exception as e:
        if is_main_process: 
            console.print(f"[red]❌ Model test failed: {e}[/red]")
        return
    training_config = final_config.get('training', {})
    if not training_config['accum_steps']:
        training_config['accum_steps'] = final_config['gradient_accumulation']
    
    trainer_config = CaptioningTrainerConfig(
        accum_steps=training_config['accum_steps'],
        max_grad_norm=training_config['max_grad_norm'],
        grad_skip_nan_inf=training_config['grad_skip_nan_inf'],
        bf16_autocast=training_config['bf16_autocast'],
        use_fp16_scaler=training_config['use_fp16_scaler'],
        val_every_steps=training_config['val_every_steps'],
        val_max_batches=training_config['val_max_batches'],
        log_every_steps=training_config['log_every_steps'],
        moving_avg_alpha=training_config['moving_avg_alpha'],
        save_every_steps=training_config['save_every_steps'],
        keep_last=training_config['keep_last'],
        async_ckpt_write=training_config['async_ckpt_write'],
        milestone_every=training_config['milestone_every'],
        use_tensorboard=training_config['use_tensorboard'],
        use_wandb=training_config['use_wandb'],
        wandb_mode=training_config['wandb_mode'],
        max_caption_length=training_config['max_caption_length'],
        generation_temperature=training_config['generation_temperature'],
        generation_top_k=training_config['generation_top_k'],
        generation_top_p=training_config['generation_top_p'],
        num_eval_samples=training_config['num_eval_samples'],
    )
    
    if is_main_process:
        console.print("[yellow]🖋️  Starting Evaluation...[/yellow]")
    yaml_string = yaml.dump(final_config, default_flow_style=False)
    if is_main_process:
        print('\n\n')
        console.print("[yellow bold]Settings:[/yellow bold]")
        print(yaml_string)
        print('\n\n')
        console.print("[yellow]🖋️  Setting up optimizer and scheduler...[/yellow]")
        console.print("[bold]GPU Setup:[/bold]")
    if args.parallel_type == 'DP':
        if is_main_process:
            print(f'Using Data Parallel ({args.parallel_type}) Setup')
        model = nn.DataParallel(model)
        model = model.to(device)
        train_sampler, val_sampler, test_sampler = None, None, None
    elif args.parallel_type == 'DDP' :
        model = model.to(device)
        if is_distributed:
            if is_main_process:
                print(f'Using Dist Data Parallel ({args.parallel_type}) Setup')
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
            train_loader, train_sampler = main_ut.get_coco_dataloader_ddp(
                tokenizer=tokenizer,
                batch_size=batch_size,
                max_seq_len=decoder_context_length+1,
                resize_to=vision_img_resize,
                datatype='train',
                num_workers=1,
                is_distributed=is_distributed,
            )

            val_loader, val_sampler = main_ut.get_coco_dataloader_ddp(
                tokenizer=tokenizer,
                batch_size=batch_size,
                max_seq_len=decoder_context_length+1,
                resize_to=vision_img_resize,
                datatype='val',
                num_workers=1,
                is_distributed=is_distributed,
            )
            
            test_loader, test_sampler = main_ut.get_coco_dataloader_ddp(
                tokenizer=tokenizer,
                batch_size=batch_size,
                max_seq_len=decoder_context_length+1,
                resize_to=vision_img_resize,
                datatype='test',
                num_workers=1,
                is_distributed=is_distributed,
            )
            
        else:
            if is_main_process:
                print(f'Parallel flag is set to ({args.parallel_type}), but failed to use as is_distributed is {is_distributed} to Use Dist Data Parallel Setup')
            model = model.to(device)
            train_sampler, val_sampler, test_sampler = None, None, None
    else:
        model = model.to(device)
        train_sampler, val_sampler, test_sampler = None, None, None
    
    optimizer = create_optimizer(model, lr=float(training_config.get('learning_rate', 1e-4)))
    max_steps = training_config.get('max_steps', 5000)
    steps_per_epoch = len(train_loader)
    total_epochs_in_steps = max(1, max_steps // steps_per_epoch)
    
    max_epochs = training_config.get('epochs', 1)
    total_steps_in_epochs = int((steps_per_epoch/ trainer_config.accum_steps) * max_epochs)
    if training_config.get('use_epochs', False):
        total_steps = total_steps_in_epochs 
    else:
        total_steps = max_steps

    scheduler = create_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=training_config.get('warmup_steps', int(0.025 * total_steps))
    )
    
    loss_fn = get_fused_cross_entropy(
        ignore_index=tokenizer.get_token_id(cnst.TOKEN_PAD)
    )
    
    
    if is_main_process:
        console.print(f"[green]✅ Training setup complete![/green]")
        console.print(f"[bold]Planned total steps:[/bold] {total_steps}")
        console.print(f"[bold]Gradient accumulation:[/bold] {trainer_config.accum_steps}")
        console.print(f"[bold]Validation every:[/bold] {trainer_config.val_every_steps} steps")
        console.print(f"[bold]Checkpoints every:[/bold] {trainer_config.save_every_steps} steps")
        console.print(f"[bold]TensorBoard:[/bold] {'✅' if trainer_config.use_tensorboard else '❌'}")
        console.print(f"[bold]W&B:[/bold] {'✅' if trainer_config.use_wandb else '❌'} (mode: {trainer_config.wandb_mode})")
    
    
    accum_steps = trainer_config.accum_steps
    save_every_steps = trainer_config.save_every_steps
    val_every_steps = trainer_config.val_every_steps

    n_params = sum(p.numel() for p in model.parameters())
    micro_steps_per_epoch = steps_per_epoch
    opt_steps_per_epoch = math.ceil(micro_steps_per_epoch / accum_steps)

    avg_overfit_time = training_config['avg_overfit_time'] if training_config['avg_overfit_time'] else 0.5
    time_per_micro_step = training_config['precomputed_train_time'] if training_config['precomputed_train_time'] > 0 else avg_overfit_time
    time_per_opt_step = time_per_micro_step * accum_steps

    num_save_events = max(1, total_steps // save_every_steps)
    save_overhead_total = num_save_events * 30

    time_one_epoch = opt_steps_per_epoch * time_per_opt_step
    time_total_run = total_steps * time_per_opt_step + save_overhead_total

    current_fixed = dt.now(pytz.timezone("Asia/Kolkata"))
    eta_run = current_fixed + datetime.timedelta(seconds=time_total_run)
    eta_one_epoch = current_fixed + datetime.timedelta(seconds=time_one_epoch)

    box0 = f"""
    [bold cyan]MODEL OVERVIEW[/bold cyan]

    • Model Type             : [bold]{args.model_type}[/bold]
    • Total Parameters       : [bold]{convert2hr2(n_params)}[/bold] ([dim]{n_params:,}[/dim])

    [bold]Architecture[/bold]
    • Encoder                : ViT
    • Captioning Head        : Transformer Decoder + LM
    • Max Caption Length     : [bold]{final_config['model'].get('max_caption_len', 64)}[/bold]
    • Vocabulary Size        : [bold]{final_config['model'].get('vocab_size', tokenizer.vocab_size)}[/bold]
    """

    box1 = f"""
    [bold cyan]DATASET OVERVIEW[/bold cyan]
    
    • Images per epoch              : [bold]{len(train_loader.dataset):,}[/bold]
    • Dataloader iterations/epoch   : [bold]{micro_steps_per_epoch:,}[/bold]
    • Accumulation steps            : [bold]{accum_steps}[/bold]
    • Optimizer steps per epoch     : [bold]{opt_steps_per_epoch:,}[/bold]
    """

    box2 = f"""
    [bold magenta]STEPS & TRAINING PLAN[/bold magenta]

    • Total planned optimizer steps   : [bold]{total_steps:,}[/bold]
    • Epochs (approx)                 : [bold]{total_steps/opt_steps_per_epoch:.2f}[/bold]
    • Save checkpoint every          : [bold]{save_every_steps}[/bold] steps (~30s overhead each)
    • Validation every               : [bold]{val_every_steps}[/bold] steps
    • Gradient accumulation          : [bold]{accum_steps}[/bold]

    • Estimated # Checkpoints         : [bold]{num_save_events}[/bold]
    • Checkpoint overhead (total)     : [bold]{fmt_time(save_overhead_total)}[/bold]
    """

    box3 = f"""
    [bold green]TIME & ETAs[/bold green]

    [bold]Per-unit times[/bold]
    • Time per micro step            : [bold]{time_per_micro_step:.3f} s[/bold]
    • Time per optimizer step        : [bold]{time_per_opt_step:.3f} s[/bold]

    [bold]Larger spans[/bold]
    • One epoch time                 : [bold]{fmt_time(time_one_epoch)}[/bold]
    • Total planned run time         : [bold]{fmt_time(time_total_run)}[/bold]

    [bold]Calendar[/bold]
    • Current time                   : [bold]{fmt_dt_ist(current_fixed)}[/bold]
    • ETA after one epoch            : [bold]{fmt_dt_ist(eta_one_epoch)}[/bold]
    • ETA end of run                 : [bold]{fmt_dt_ist(eta_run)}[/bold]
    """
    if is_main_process:
        print("\n\n")
        console.print(Panel.fit(box0, title="🧠 Model", border_style="yellow", box=HEAVY))
        # console.print(Panel.fit(box1, title="📦 Dataset", border_style="cyan", box=HEAVY))
        # console.print(Panel.fit(box2, title="🧮 Steps & Plan", border_style="magenta", box=HEAVY))
        # console.print(Panel.fit(box3, title="⏱️ Time Estimates", border_style="green", box=HEAVY))
        print("\n\n")
        
        console.print("\n[yellow]🖋️ Initializing trainer...[/yellow]")
    

    if args.test:
        if is_main_process:
            console.rule("[bold green]🧪 Starting Evaluation[/bold green]")
            console.print(f"Using model from run: [bold]{training_config['run_name']}[/bold]")
        
        # Ensure model on correct device
        model = model.to(device)
        model.eval()

        # Reload best checkpoint (if exists)
        ckpt_dir = Path(training_config['log_dir']) / args.model_type / training_config['run_name'] / 'dist'
        ckpt_model_path = ckpt_dir / 'model.pt'
        if ckpt_model_path.exists():
            state_dict = torch.load(ckpt_model_path, map_location=device)
            model.load_state_dict(state_dict)
            if is_main_process:
                console.print(f"[cyan]✅ Loaded best checkpoint from:[/cyan] {ckpt_model_path}")
        else:
            if is_main_process:
                console.print(f"[yellow]⚠️ No best checkpoint found at {ckpt_model_path}, using current weights.[/yellow]")

        # Initialize Trainer for evaluation utilities
        eval_trainer = ImageCaptioningTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_loader=train_loader,
            val_loader=test_loader,
            tokenizer=tokenizer,
            device=device,
            model_type=args.model_type,
            run_name=training_config['run_name'],
            log_dir=training_config['log_dir'],
            scheduler=scheduler,
            config=trainer_config,
            model_kwargs=model_kwargs,
            is_distributed=is_distributed,
            train_sampler=train_sampler,
            val_sampler=test_sampler,
            verbose=True
        )

        # Run evaluation (set max_batches=None to use full val dataset)
        max_batches = trainer_config.val_max_batches if trainer_config.val_max_batches else None
        if is_main_process:
            console.print(f"[yellow]🖋 Evaluating on {max_batches or 'ALL'} validation batches...[/yellow]")
        val_metrics = eval_trainer.evaluate(max_batches=max_batches)

        if is_main_process:
            console.print(f"\n[bold green]✅ Evaluation Completed![/bold green]")
            console.print(f"📉 [bold]Val Loss:[/bold] {val_metrics.get('loss', 0):.4f}")
            console.print(f"📈 [bold]BLEU:[/bold] {val_metrics.get('bleu', 0):.4f}")
            console.print(f"📜 [bold]ROUGE-L:[/bold] {val_metrics.get('rouge_l', 0):.4f}")
            console.print(f"📊 [bold]Perplexity:[/bold] {val_metrics.get('perplexity', 0):.4f}")

            # 🔹 Save metrics
            results_dir = Path(training_config['log_dir']) / args.model_type / training_config['run_name'] / 'eval_results'
            results_dir.mkdir(parents=True, exist_ok=True)
            results_file = results_dir / 'evaluation_results.json'
            with open(results_file, 'w') as f:
                json.dump(val_metrics, f, indent=4)
            console.print(f"[cyan]📁 Metrics saved to:[/cyan] {results_file}")

            # 🔹 Generate a few qualitative samples
            console.print("\n[yellow]🖋 Generating qualitative samples...[/yellow]")
            eval_trainer.save_sample_results(step='final')

            console.print(f"[green]✅ Sample captions saved in:[/green] {eval_trainer.dirs['samples']}")
            console.rule("[bold green]Evaluation Summary Complete[/bold green]")

if __name__ == "__main__":
    console.print('\n[orange]' + '-'*80 + '[/orange]' )
    console.print('[bold green]\t\t👋 Hi! \t Image-Captioning (ViT-GPT) Train/Eval Arena [/bold green]')
    console.print('[orange]' + '-'*80 + '[/orange]\n\n' )
    start = dt.now()
    main()
    console.print('[bold green]✅ Done! Script Completed Successfuly[/bold green]')
    print(f'⌚ Script Time: {dt.now() - start}')