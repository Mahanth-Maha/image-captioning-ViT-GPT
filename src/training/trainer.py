import os
import json
import math
import time
import signal
import shutil
import threading
from datetime import datetime
from pathlib import Path
from contextlib import nullcontext
from dataclasses import dataclass

import constants as cnst

import torch
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist



import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from torch.utils.tensorboard import SummaryWriter
import wandb

@dataclass
class CaptioningTrainerConfig:
    accum_steps: int = 1
    max_grad_norm: float = 1.0
    grad_skip_nan_inf: bool = True
    bf16_autocast: bool = True
    use_fp16_scaler: bool = False
    
    val_every_steps: int = 1000
    val_max_batches: int = 100
    log_every_steps: int = 50
    moving_avg_alpha: float = 0.03

    save_every_steps: int = 5000
    keep_last: int = 3
    async_ckpt_write: bool = True
    milestone_every: int = 10000

    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_mode: str = "offline"  # "online", "offline", "disabled"

    max_caption_length: int = 35
    generation_temperature: float = 0.9
    generation_top_k: int = 50
    generation_top_p: float = 0.9
    num_eval_samples: int = 5


@torch.no_grad()
def reduce_tensor(tensor, average=True):
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if average:
        tensor /= dist.get_world_size()
    return tensor

class ImageCaptioningTrainer:
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader, tokenizer, 
                 device=None, model_type="vit_base_gpt", run_name=None, log_dir="experiments", 
                 scheduler=None, config=None, model_kwargs=None, 
                 is_distributed = False, train_sampler=None, val_sampler=None,
                 verbose= 'info', save_image_results = False):
        
        self.is_distributed = is_distributed
        if isinstance(verbose, bool):
            if verbose:
                self.verbose = 'info'
            else:
                self.verbose = 'silent'
        else: 
            self.verbose = verbose
        self.model_type = model_type
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
            
        if self.verbose == 'info':
            print(f"🖥️ Training on the device: {self.device}")
            
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.tokenizer = tokenizer
        self.model_kwargs = model_kwargs or {}
        self.save_image_results = save_image_results
        
        self.config = config or CaptioningTrainerConfig()
        
        # self.scaler = torch.amp.GradScaler() if self.config.bf16_autocast and self.device.type == 'cuda' else None
        # self.device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
        if not hasattr(self.config, "use_fp16_scaler"):
            self.config.use_fp16_scaler = False
        bf16_supported = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
        if self.config.bf16_autocast and not bf16_supported:
            print("⚠️ bfloat16 not supported — falling back to fp16 autocast.")
            self.config.bf16_autocast = False
            self.config.use_fp16_scaler = True

        # New unified torch.amp API
        self.device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
        self.scaler = (
            torch.amp.GradScaler() if self.config.use_fp16_scaler else None
        )
        
        self.setup_directories(log_dir)
        
        self.pad_token_id = tokenizer.special_tokens.get(cnst.TOKEN_PAD, -100)
        self.sos_token_id = tokenizer.special_tokens.get(cnst.TOKEN_SOS, 0)
        self.eos_token_id = tokenizer.special_tokens.get(cnst.TOKEN_EOS, 1)
        
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_bleu = 0.0
        self.checkpoint_history = []
        
        self.logs = {
            "train_loss": [], "train_loss_ma": [], "val_loss": [], "val_bleu": [],
            "val_rouge_l": [], "val_cider": [], "steps_train": [], "steps_val": [],
            "lr": [], "tok_per_sec": [], "grad_norm": [], "epochs": [], 'step_time':[]
        }
        if self.is_main_process():
            self.setup_loggers()
            self.setup_interrupt_handler()

        if self.verbose == 'debug':
            print(f"Trainer initialized for {self.model_type}/{self.run_name}")
            
    def setup_directories(self, log_dir):
        self.base_dir = Path(log_dir) / self.model_type / self.run_name
        
        self.dirs = {
            'base': self.base_dir,
            'dist': self.base_dir / 'dist',
            'training': self.base_dir / 'training',
            'results': self.base_dir / 'results',
            'checkpoints': self.base_dir / 'training' / 'checkpoints',
            'milestones': self.base_dir / 'training' / 'milestones',
            'interrupt': self.base_dir / 'training' / 'interrupt',
            'metrics': self.base_dir / 'training' / 'metrics',
            'plots': self.base_dir / 'results' / 'plots',
            'samples': self.base_dir / 'results' / 'samples',
            'tensorboard': Path(log_dir) / self.model_type / 'tensorboard' / self.run_name,
            'wandb': Path(log_dir) / self.model_type / 'wandb' / self.run_name
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def setup_loggers(self):
        self.tb_writer = None
        if self.config.use_tensorboard:
            self.tb_writer = SummaryWriter(log_dir=str(self.dirs['tensorboard']))
            self.tb_writer.add_text("run_info", f"Model: {self.model_type}, Run: {self.run_name}")
            if self.verbose == 'info':
                print(f"🪵 TensorBoard logging to: {self.dirs['tensorboard']}")
                
        self.wandb_run = None
        if self.config.use_wandb:
            wandb_dir = str(self.dirs['wandb'])
            wandb.init(
                project=f"image_captioning_{self.model_type}",
                name=self.run_name,
                dir=wandb_dir,
                mode=self.config.wandb_mode,
                config={
                    'model_type': self.model_type,
                    'model_params': sum(p.numel() for p in self.model.parameters()),
                    'device': str(self.device),
                    **self.model_kwargs,
                    **self.config.__dict__
                }
            )
            self.wandb_run = wandb
            if self.verbose == 'info':
                print(f"🪵 W&B logging initialized (mode: {self.config.wandb_mode})")
                
    def setup_interrupt_handler(self):
        def signal_handler(signum, frame):
            if self.verbose == 'info':
                print(f"\n🛑 Received signal {signum}. Saving interrupt checkpoint...")
            if self.is_main_process():
                self.save_interrupt_checkpoint()
            exit(1)
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
    def log_metrics(self, metrics_dict, step, prefix=""):
        if self.is_main_process():
            if prefix:
                log_str = f"{prefix} - Step {step}: "
            else:
                log_str = f"Step {step}: "
            log_str += " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics_dict.items()])
            if self.verbose == 'debug':
                print(log_str)
                
            if self.tb_writer:
                for key, value in metrics_dict.items():
                    if isinstance(value, (int, float)):
                        tag = f"{prefix}/{key}" if prefix else key
                        self.tb_writer.add_scalar(tag, value, step)
                        
            if self.wandb_run:
                wandb_dict = {f"{prefix}/{key}" if prefix else key: value 
                            for key, value in metrics_dict.items() 
                            if isinstance(value, (int, float))}
                wandb_dict['step'] = step
                self.wandb_run.log(wandb_dict)
                
    def prepare_batch(self, batch):
        images, captions = batch
        # if self.is_distributed:
        images = images.to(self.device, memory_format=torch.channels_last, non_blocking=True)
        captions = captions.to(self.device, non_blocking=True)
        # else:
        #     images = images.to(self.device)
        #     captions = captions.to(self.device)

        return images, captions
        
    def train_step(self, batch):
        self.model.train()
        images, captions = self.prepare_batch(batch)
        
        # ctx = torch.autocast(self.device_type, dtype=torch.bfloat16) if self.config.bf16_autocast else nullcontext()
        amp_dtype = torch.bfloat16 if self.config.bf16_autocast else torch.float16
        ctx = torch.amp.autocast(device_type=self.device_type, dtype=amp_dtype) if self.device_type == 'cuda' else nullcontext()

        with ctx:
            output_logits = self.model(images, captions[:, :-1])
            labels = captions[:, 1:]
            logits_flat = output_logits.reshape(-1, output_logits.size(-1))
            labels_flat = labels.reshape(-1)
            loss = self.loss_fn(logits_flat, labels_flat) / self.config.accum_steps
            
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
            
        # return loss.item() * self.config.accum_steps
        loss_value = loss.item() * self.config.accum_steps

        if self.is_distributed:
            loss_tensor = torch.tensor([loss_value], device=self.device)
            loss_value = reduce_tensor(loss_tensor).item()
        return loss_value
        
    def optimizer_step(self):
        grad_norm = None
        if self.config.max_grad_norm > 0:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            grad_norm = clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm).item()
            
        if self.config.grad_skip_nan_inf:
            has_bad_grad = any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in self.model.parameters()
            )
            if has_bad_grad:
                if self.verbose == 'info':
                    print("⚠️ Skipping step due to NaN/Inf gradients")
                self.optimizer.zero_grad()
                return False, grad_norm
                
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
            
        self.optimizer.zero_grad()
        if self.scheduler:
            self.scheduler.step()
            
        return True, grad_norm
    
    def is_main_process(self):
        return (not self.is_distributed) or (dist.get_rank() == 0)
    
    @torch.no_grad()
    def evaluate(self, max_batches=None):
        self.model.eval()
        max_batches = max_batches or self.config.val_max_batches
        
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_references = []
        
        val_iter = iter(self.val_loader)
        for _ in tqdm(range(max_batches), desc="Validation", leave=False, disable=not self.is_main_process()):
            try:
                batch = next(val_iter)
            except StopIteration:
                break
                
            images, captions = self.prepare_batch(batch)
            
            ctx = torch.autocast(self.device_type, dtype=torch.bfloat16) if self.config.bf16_autocast else nullcontext()
            with ctx:
                output_logits = self.model(images, captions[:, :-1])
                labels = captions[:, 1:]
                logits_flat = output_logits.reshape(-1, output_logits.size(-1))
                labels_flat = labels.reshape(-1)
                loss = self.loss_fn(logits_flat, labels_flat)
            
                
            total_loss += loss.item()
            num_batches += 1
            
            
            if num_batches <= 5:
                if self.is_distributed:
                    generated = self.model.module.generate_caption(
                        images=images[:2],
                        tokenizer=self.tokenizer,
                        max_length=self.config.max_caption_length,
                        temperature=self.config.generation_temperature,
                        top_k=self.config.generation_top_k,
                        top_p=self.config.generation_top_p
                    )
                else:
                    generated = self.model.generate_caption(
                        images=images[:2],
                        tokenizer=self.tokenizer,
                        max_length=self.config.max_caption_length,
                        temperature=self.config.generation_temperature,
                        top_k=self.config.generation_top_k,
                        top_p=self.config.generation_top_p
                    )
                
                for i in range(min(2, generated.shape[0])):
                    pred_tokens = generated[i].cpu().tolist()
                    pred_tokens = [t for t in pred_tokens if t not in [self.sos_token_id, self.eos_token_id, self.pad_token_id]]
                    pred_caption = self.tokenizer.decode(pred_tokens)
                    
                    ref_tokens = labels[i].cpu().tolist()
                    # ref_tokens = [t for t in ref_tokens if t >= 0]  # Remove -100 (padding)
                    ref_caption = self.tokenizer.decode(ref_tokens)
                    
                    all_predictions.append(pred_caption.split())
                    all_references.append([ref_caption.split()])
                    
        avg_loss = total_loss / max(num_batches, 1)
        if self.is_distributed:
            avg_loss_tensor = torch.tensor(avg_loss, device=self.device)
            avg_loss = reduce_tensor(avg_loss_tensor).item()
        metrics = {'loss': avg_loss, 'perplexity': math.exp(min(avg_loss, 10))}
        
        if all_predictions:
            # BLEU score
            smoothing = SmoothingFunction().method4
            bleu_scores = [
                sentence_bleu(ref, pred, smoothing_function=smoothing)
                for ref, pred in zip(all_references, all_predictions)
            ]
            metrics['bleu'] = np.mean(bleu_scores) if bleu_scores else 0.0
            
            # ROUGE-L score
            rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            rouge_scores = [
                rouge.score(' '.join(ref[0]), ' '.join(pred))['rougeL'].fmeasure
                for ref, pred in zip(all_references, all_predictions)
            ]
            metrics['rouge_l'] = np.mean(rouge_scores) if rouge_scores else 0.0
        
        if self.is_distributed and all_predictions:
            bleu_tensor = torch.tensor(metrics['bleu'], device=self.device)
            rouge_tensor = torch.tensor(metrics['rouge_l'], device=self.device)
            metrics['bleu'] = reduce_tensor(bleu_tensor).item()
            metrics['rouge_l'] = reduce_tensor(rouge_tensor).item()
        
        return metrics
        
    def save_sample_results(self, step):
        if self.is_main_process():
            self.model.eval()
            
            try:
                val_batch = next(iter(self.val_loader))
                images, captions = val_batch
                
                num_samples = min(self.config.num_eval_samples, images.shape[0])
                sample_images = images[:num_samples].to(self.device)
                sample_captions = captions[:num_samples]
                
                with torch.no_grad():
                    if self.is_distributed:
                        generated = self.model.module.generate_caption(
                            images=sample_images,
                            tokenizer=self.tokenizer,
                            max_length=self.config.max_caption_length,
                            temperature=0.7,
                            top_k=self.config.generation_top_k
                        )
                    else:
                        generated = self.model.generate_caption(
                            images=sample_images,
                            tokenizer=self.tokenizer,
                            max_length=self.config.max_caption_length,
                            temperature=0.7,
                            top_k=self.config.generation_top_k
                        )
                
                if self.save_image_results:
                    results_dir = self.dirs['samples'] / f"step_{step}"
                    results_dir.mkdir(exist_ok=True)
                    
                    for i in range(num_samples):
                        gen_tokens = generated[i].cpu().tolist()
                        generated_caption = self.tokenizer.decode(gen_tokens)
                        
                        ref_tokens = sample_captions[i].tolist()
                        ref_tokens = [t for t in ref_tokens if t != self.pad_token_id]
                        reference_caption = self.tokenizer.decode(ref_tokens)
                        
                        img_tensor = sample_images[i].cpu()
                        mean = torch.tensor([0.471, 0.448, 0.408]).view(3, 1, 1)
                        std = torch.tensor([0.234, 0.239, 0.242]).view(3, 1, 1)
                        img_tensor = img_tensor * std + mean
                        img_tensor = torch.clamp(img_tensor, 0, 1)
                        
                        img_pil = Image.fromarray((img_tensor.permute(1, 2, 0) * 255).numpy().astype(np.uint8))
                        
                        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                        ax.imshow(img_pil)
                        ax.axis('off')
                        ax.set_title(f"Generated: {generated_caption}\nReference: {reference_caption}", 
                                wrap=True, fontsize=10)
                        
                        plt.tight_layout()
                        plt.savefig(results_dir / f"sample_{i}.png", dpi=150, bbox_inches='tight')
                        plt.close()
                        
                        with open(results_dir / f"sample_{i}.txt", 'w') as f:
                            f.write(f"Step: {step}\n")
                            f.write(f"Generated: {generated_caption}\n")
                            f.write(f"Reference: {reference_caption}\n")
                else:
                    results_dir = self.dirs['samples']
                    results_dir.mkdir(exist_ok=True)
                    results_file = results_dir / "samples.md"

                    for i in range(num_samples):
                        gen_tokens = generated[i].cpu().tolist()
                        generated_caption = self.tokenizer.decode(gen_tokens)
                        
                        ref_tokens = sample_captions[i].tolist()
                        ref_tokens = [t for t in ref_tokens if t != self.pad_token_id]
                        reference_caption = self.tokenizer.decode(ref_tokens)
                        
                        with open(results_file, 'a') as f:
                            if i == 0:
                                f.write(f"\n## Checkpoint {step}\n\n")
                            
                            f.write(f"### Sample {i + 1}\n\n")
                            f.write(f"**Reference:** \n```\n{reference_caption}\n```\n")
                            f.write(f"**Generated:** \n```\n{generated_caption}\n```\n\n")
                        
            except Exception as e:
                if self.verbose== 'info':
                    print(f"⚠️ Failed to save sample results: {e}")
                    
    def save_plots(self, step):
        if self.is_main_process():
            plots_dir = self.dirs['plots']
            
            if self.logs['steps_train'] and self.logs['train_loss']:
                plt.figure(figsize=(10, 6))
                plt.plot(self.logs['steps_train'], self.logs['train_loss'], label='Train Loss', alpha=0.7)
                if self.logs['train_loss_ma']:
                    plt.plot(self.logs['steps_train'], self.logs['train_loss_ma'], label='Train Loss (MA)', linewidth=2)
                if self.logs['steps_val'] and self.logs['val_loss']:
                    plt.plot(self.logs['steps_val'], self.logs['val_loss'], label='Val Loss', marker='o')
                plt.xlabel('Steps')
                plt.ylabel('Loss')
                plt.title('Training and Validation Loss')
                plt.legend()
                plt.grid(True, alpha=0.3)
                # plt.savefig(plots_dir / f'loss_step_{step}.png', dpi=150, bbox_inches='tight')
                plt.savefig(plots_dir / f'loss_step.png', dpi=150, bbox_inches='tight')
                plt.close()
                
            if self.logs['steps_val'] and self.logs['val_bleu']:
                plt.figure(figsize=(10, 6))
                plt.plot(self.logs['steps_val'], self.logs['val_bleu'], label='BLEU Score', marker='o', color='green')
                plt.xlabel('Steps')
                plt.ylabel('BLEU Score')
                plt.title('Validation BLEU Score')
                plt.legend()
                plt.grid(True, alpha=0.3)
                # plt.savefig(plots_dir / f'bleu_step_{step}.png', dpi=150, bbox_inches='tight')
                plt.savefig(plots_dir / f'bleu_step.png', dpi=150, bbox_inches='tight')
                plt.close()
                
            if self.logs['steps_train'] and self.logs['lr']:
                plt.figure(figsize=(10, 6))
                plt.plot(self.logs['steps_train'], self.logs['lr'], label='Learning Rate')
                plt.xlabel('Steps')
                plt.ylabel('Learning Rate')
                plt.title('Learning Rate Schedule')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.yscale('log')
                # plt.savefig(plots_dir / f'lr_step_{step}.png', dpi=150, bbox_inches='tight')
                plt.savefig(plots_dir / f'lr_step.png', dpi=150, bbox_inches='tight')
                plt.close()
            
    def save_checkpoint(self, step, is_best=False, is_milestone=False, is_interrupt=False):
        if self.is_main_process():
            if is_interrupt:
                ckpt_dir = self.dirs['interrupt']
                ckpt_name = f"interrupt_step_{step}"
            elif is_milestone:
                ckpt_dir = self.dirs['milestones']
                ckpt_name = f"milestone_step_{step}"
            else:
                ckpt_dir = self.dirs['checkpoints']
                ckpt_name = f"checkpoint_step_{step}"
                
            ckpt_path = ckpt_dir / ckpt_name
            ckpt_path.mkdir(exist_ok=True)
            
            torch.save(self.model.state_dict(), ckpt_path / "model.pt")
            if self.model_kwargs:
                torch.save(self.model_kwargs, ckpt_path / "model_config.pt")
            training_state = {
                'optimizer_state': self.optimizer.state_dict(),
                'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
                'scaler_state': self.scaler.state_dict() if self.scaler else None,
                'step': self.global_step,
                'epoch': self.current_epoch,
                'best_val_loss': self.best_val_loss,
                'best_bleu': self.best_bleu,
                'config': self.config.__dict__,
                'logs': self.logs
            }
            torch.save(training_state, ckpt_path / "training_state.pt")
            with open(ckpt_path / "logs.json", 'w') as f:
                json.dump(self.logs, f, indent=2)
                
            if not is_interrupt and is_milestone:
                self.save_plots(step)
                if step > 0:
                    self.save_sample_results(step)
                    
            if is_best:
                dist_path = self.dirs['dist']
                if dist_path.exists():
                    shutil.rmtree(dist_path)
                shutil.copytree(ckpt_path, dist_path)
                
            if not is_milestone and not is_interrupt:
                self.checkpoint_history.append(ckpt_path)
                while len(self.checkpoint_history) > self.config.keep_last:
                    old_ckpt = self.checkpoint_history.pop(0)
                    if old_ckpt.exists():
                        shutil.rmtree(old_ckpt)
                        
            if self.verbose == 'debug':
                checkpoint_type = "interrupt" if is_interrupt else ("milestone" if is_milestone else "regular")
                print(f"💾 Saved {checkpoint_type} checkpoint: {ckpt_path}")
                
            return ckpt_path
        
    def save_interrupt_checkpoint(self):
        if self.is_main_process():
            check_pt = self.save_checkpoint(self.global_step, is_interrupt=True)
        return check_pt
        
    def find_latest_checkpoint(self):
        interrupt_dir = self.dirs['interrupt']
        if interrupt_dir.exists():
            interrupt_ckpts = [d for d in interrupt_dir.iterdir() if d.is_dir()]
            if interrupt_ckpts:
                latest_interrupt = max(interrupt_ckpts, 
                                     key=lambda x: int(x.name.split('_')[-1]) if x.name.split('_')[-1].isdigit() else 0)
                return latest_interrupt
                
        milestone_dir = self.dirs['milestones']
        milestone_ckpts = []
        if milestone_dir.exists():
            milestone_ckpts = [d for d in milestone_dir.iterdir() if d.is_dir()]
            
        checkpoint_dir = self.dirs['checkpoints']
        regular_ckpts = []
        if checkpoint_dir.exists():
            regular_ckpts = [d for d in checkpoint_dir.iterdir() if d.is_dir()]
            
        all_ckpts = milestone_ckpts + regular_ckpts
        if not all_ckpts:
            return None
            
        latest_ckpt = max(all_ckpts, 
                         key=lambda x: int(x.name.split('_')[-1]) if x.name.split('_')[-1].isdigit() else 0)
        return latest_ckpt
        
    def resume_training(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = self.find_latest_checkpoint()
            
        if checkpoint_path is None:
            if self.verbose == 'info':
                print("No checkpoint found to resume from.")
            return False
            
        if self.verbose == 'info':
            print(f"Resuming from checkpoint: {checkpoint_path}")
            
        model_path = checkpoint_path / "model.pt"
        if model_path.exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        training_state_path = checkpoint_path / "training_state.pt"
        if training_state_path.exists():
            state = torch.load(training_state_path, map_location=self.device)
            
            self.global_step = state.get('step', 0)
            self.current_epoch = state.get('epoch', 0)
            self.best_val_loss = state.get('best_val_loss', float('inf'))
            self.best_bleu = state.get('best_bleu', 0.0)
            self.logs = state.get('logs', self.logs)
            if 'optimizer_state' in state:
                self.optimizer.load_state_dict(state['optimizer_state'])
            if self.scheduler and 'scheduler_state' in state and state['scheduler_state']:
                self.scheduler.load_state_dict(state['scheduler_state'])
            if self.scaler and 'scaler_state' in state and state['scaler_state']:
                self.scaler.load_state_dict(state['scaler_state'])
                
        if self.verbose == 'info':
            print(f"✅ Resumed from step {self.global_step}, epoch {self.current_epoch}")
            
        return True
        
    def train(self, max_steps=50000, auto_resume=True, is_warmup = False):
        rank = dist.get_rank() if self.is_distributed and dist.is_initialized() else 0
        world_size = dist.get_world_size() if self.is_distributed and dist.is_initialized() else 1
        # is_main_process = (rank == 0)
        if self.is_distributed:
            torch.cuda.set_device(rank % torch.cuda.device_count())
            
        if auto_resume and not is_warmup:
            self.resume_training()
            
        if self.is_main_process() and self.verbose == 'info':
            print(f" Starting training from step {self.global_step} to {max_steps}")
        
        if self.is_distributed and self.train_sampler is not None:
            self.train_sampler.set_epoch(self.current_epoch)
        
        self.model.train()
        train_iter = iter(self.train_loader)
        
        loss_ma = None
        tokens_per_step = 0
        step_start_time = time.time()
        
        # with tqdm(initial=self.global_step, total=max_steps, desc="Training") as pbar:
        with tqdm(initial=self.global_step, total=max_steps, desc="Training", disable=not self.is_main_process()) as pbar:

            while self.global_step < max_steps:
                try:
                    batch = next(train_iter)
                except StopIteration:
                    self.current_epoch += 1
                    if self.is_distributed and self.train_sampler is not None:
                        self.train_sampler.set_epoch(self.current_epoch)
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)
                    
                step_loss = 0.0
                tokens_this_step = 0
                
                for micro_step in range(self.config.accum_steps):
                    if micro_step > 0:
                        try:
                            batch = next(train_iter)
                        except StopIteration:
                            self.current_epoch += 1
                            if self.is_distributed and self.train_sampler is not None:
                                self.train_sampler.set_epoch(self.current_epoch)
                            train_iter = iter(self.train_loader)
                            batch = next(train_iter)
                            
                    loss = self.train_step(batch)
                    step_loss += loss
                    tokens_this_step += batch[1].numel()
                    
                success, grad_norm = self.optimizer_step()
                if not success:
                    continue
                    
                self.global_step += 1
                
                if loss_ma is None:
                    loss_ma = step_loss
                else:
                    loss_ma = (1 - self.config.moving_avg_alpha) * loss_ma + self.config.moving_avg_alpha * step_loss
                    
                step_time = time.time() - step_start_time
                tokens_per_sec = tokens_this_step / max(step_time, 1e-6)
                step_start_time = time.time()
                
                lr = self.optimizer.param_groups[0]['lr']
                self.logs['train_loss'].append(step_loss)
                self.logs['train_loss_ma'].append(loss_ma)
                self.logs['steps_train'].append(self.global_step)
                self.logs['lr'].append(lr)
                self.logs['tok_per_sec'].append(tokens_per_sec)
                self.logs['step_time'].append(step_time)
                self.logs['grad_norm'].append(grad_norm)
                self.logs['epochs'].append(self.current_epoch)
                
                if self.global_step % self.config.log_every_steps == 0:
                    metrics = {
                        'train_loss': step_loss,
                        'train_loss_ma': loss_ma,
                        'lr': lr,
                        'tokens_per_sec': tokens_per_sec,
                        'epoch': self.current_epoch
                    }
                    if grad_norm is not None:
                        metrics['grad_norm'] = grad_norm
                    self.log_metrics(metrics, self.global_step, "train")
                    
                if self.global_step % self.config.val_every_steps == 0:
                    val_metrics = self.evaluate()
                    
                    self.logs['val_loss'].append(val_metrics['loss'])
                    self.logs['steps_val'].append(self.global_step)
                    
                    if 'bleu' in val_metrics:
                        self.logs['val_bleu'].append(val_metrics['bleu'])
                    if 'rouge_l' in val_metrics:
                        self.logs['val_rouge_l'].append(val_metrics['rouge_l'])
                        
                    self.log_metrics(val_metrics, self.global_step, "val")
                    
                    is_best = False
                    if val_metrics['loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['loss']
                        is_best = True
                        
                    if 'bleu' in val_metrics and val_metrics['bleu'] > self.best_bleu:
                        self.best_bleu = val_metrics['bleu']
                        is_best = True
                        
                    if self.is_main_process() and is_best and not is_warmup:
                        self.save_checkpoint(self.global_step, is_best=True)
                        
                if self.is_main_process() and self.global_step % self.config.save_every_steps == 0:
                    self.save_checkpoint(self.global_step)
                    
                if self.is_main_process() and self.global_step % self.config.milestone_every == 0 and self.global_step > 0 and not is_warmup:
                    self.save_checkpoint(self.global_step, is_milestone=True)
                    
                pbar.set_postfix({
                    'loss': f'{step_loss:.4f}',
                    'loss_ma': f'{loss_ma:.4f}',
                    'val_loss': f'{self.logs["val_loss"][-1]:.4f}' if self.logs['val_loss'] else 'N/A',
                    'bleu': f'{self.logs["val_bleu"][-1]:.3f}' if self.logs['val_bleu'] else 'N/A',
                    'lr': f'{lr:.2e}',
                    'tok/s': f'{tokens_per_sec:.0f}',
                    'epoch': self.current_epoch
                })
                pbar.update(1)
        if self.is_main_process() and not is_warmup:
            self.save_checkpoint(self.global_step, is_best=True)
        
        if self.is_distributed and dist.is_initialized():
            dist.barrier()
            if dist.get_rank() == 0:
                print("🧹 DDP synchronization barrier reached — all ranks finished training.")

        if self.is_main_process(): 
            if self.verbose == 'info':
                print(f"🎉 Training completed! Best val loss: {self.best_val_loss:.4f}, Best BLEU: {self.best_bleu:.3f}")
            if self.tb_writer:
                self.tb_writer.close()
            if self.wandb_run:
                self.wandb_run.finish()
            
