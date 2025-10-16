import torch
import random
import numpy as np
from training.cnn_lstm_trainer import CNNLSTMConfig, CNNLSTMTrainer
from model.cnn_lstm_img_cap_model import CNNLSTMCaptioner
from utils import get_coco_dataloader, get_tokenizer
import constants as cnst
import argparse
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime 

from rich.box import HEAVY
from rich.panel import Panel
from rich.console import Console
console = Console()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_args():
    parser = argparse.ArgumentParser(description="Train CNN+LSTM Captioning Model on COCO")
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--resize_to", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    
    parser.add_argument("--cnn_name", type=str, default=None)
    parser.add_argument("--cnn_train_backbone", action="store_true", help="Enable CNN backbone training")
    parser.add_argument("--cnn_out_dim", type=int, default=None)
    parser.add_argument("--proj_dim", type=int, default=None)
    parser.add_argument("--embed_dim", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    
    parser.add_argument("--overfit_test", action="store_true", help="Run single-batch overfit test before training")
    parser.add_argument("--overfit_test_steps", type=int, default=50)
    parser.add_argument("--lr_sweep_test", action="store_true", help="Run LR sweep test before training")
    parser.add_argument("--lr_sweep_test_steps", type=int, default=50)

    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--amp", action="store_true", help="Enable AMP mixed precision training")

    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--samples_per_epoch", type=int, default=None)
    parser.add_argument("--print_every", type=int, default=None)
    parser.add_argument("--sample_every", type=int, default=None)

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)

    return parser.parse_args()


def update_config_from_args(cfg, args):
    for key, value in vars(args).items():
        if value is not None:
            setattr(cfg, key, value)
    return cfg




def overfit_single_batch_test(cfg, tokenizer, train_loader, max_steps = 250):
    print("\n[Overfit Test] Starting single-batch overfit sanity check...")
    start_time = datetime.datetime.now()
    os.makedirs(os.path.join(cfg.out_dir, "diagnostics"), exist_ok=True)

    pad_idx = tokenizer.special_tokens[cnst.TOKEN_PAD]
    test_model = CNNLSTMCaptioner(vocab_size=cfg.vocab_size, cfg=cfg, pad_idx=pad_idx).to(cfg.device)
    test_model.train()

    images, captions = next(iter(train_loader))
    images, captions = images.to(cfg.device), captions.to(cfg.device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(test_model.parameters(), lr=cfg.lr)

    losses = []
    for step in tqdm(range(max_steps), desc='Ovefitting Steps'):
        optimizer.zero_grad()
        logits = test_model(images, captions)
        loss = criterion(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            captions[:, 1:].reshape(-1)
        )
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        # if (step + 1) % 20 == 0:
        #     print(f"[Overfit] Step {step+1}/{max_steps} | Loss: {loss.item():.4f}")

    plt.figure(figsize=(6, 4))
    plt.plot(losses, label='Training Loss', color='blue')
    plt.title("Overfit Test: Loss vs. Steps")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    save_path = os.path.join(cfg.out_dir, "diagnostics", "overfit_single_batch.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"✅ Final Overfit Loss: {losses[-1]:.4f}")
    end_time = datetime.datetime.now()
    print(f"Time Took: {end_time - start_time}")
    print(f"Loss curve saved to: {save_path}")
    return losses


def lr_sweeping_test(cfg, tokenizer, lrs=[1e-6, 1e-5, 5e-5, 1e-4, 3e-4, 5e-4, 7e-4, 1e-3, 3e-3, 5e-3, 7e-3, 1e-2], steps=100):
    print("\n⚙️ [LR Sweep] Testing candidate learning rates:", lrs)
    os.makedirs(os.path.join(cfg.out_dir, "diagnostics"), exist_ok=True)

    pad_idx = tokenizer.special_tokens[cnst.TOKEN_PAD]

    lr_results = {}

    for lr in lrs:
        start_time = datetime.datetime.now()
        model = CNNLSTMCaptioner(vocab_size=cfg.vocab_size, cfg=cfg, pad_idx=pad_idx).to(cfg.device)
        model.train()

        train_loader = get_coco_dataloader(
            tokenizer=tokenizer,
            batch_size=cfg.batch_size,
            max_seq_len=cfg.max_seq_len,
            resize_to=cfg.resize_to,
            datatype='train'
        )
        
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
        optimizer = optim.AdamW(model.parameters(), lr=lr)

        total_loss = 0.0
        print(f"\n🔹 Testing LR = {lr:.1e}")
        for step in tqdm(range(steps), desc=f"LR {lr:.1e}", leave=False):
            images, captions = next(iter(train_loader))
            images, captions = images.to(cfg.device), captions.to(cfg.device)
            optimizer.zero_grad()
            logits = model(images, captions)
            loss = criterion(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                captions[:, 1:].reshape(-1)
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / steps
        lr_results[lr] = avg_loss
        print(f"▶️  Avg Loss @ LR={lr:.1e}: {avg_loss:.4f}")
        del model 
        torch.cuda.empty_cache()
        end_time = datetime.datetime.now()
        print(f"⏲️  Time Took: {end_time - start_time}")

    sorted_lrs = sorted(lr_results.keys())
    sorted_losses = [lr_results[lr] for lr in sorted_lrs]

    plt.figure(figsize=(7, 5))
    plt.semilogx(sorted_lrs, sorted_losses, marker='o', color='orange', linewidth=2)
    plt.title("LR Sweep: Average Loss vs. Learning Rate")
    plt.xlabel("Learning Rate (log scale)")
    plt.ylabel("Average Loss")
    plt.grid(True, linestyle='--', alpha=0.6)

    best_lr = min(lr_results, key=lr_results.get)
    plt.axvline(best_lr, color='green', linestyle='--', label=f"Best LR = {best_lr:.2e}")
    plt.legend()

    save_path = os.path.join(cfg.out_dir, "diagnostics", "lr_sweep.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"\n🏆 Best LR Found: {best_lr:.2e} with avg loss {lr_results[best_lr]:.4f}")
    print(f"LR sweep plot saved to: {save_path}")

    return best_lr, lr_results



def main():
    args = get_args()
    cfg = CNNLSTMConfig()
    cfg = update_config_from_args(cfg, args)
    set_seed(cfg.seed)

    print("Loading tokenizer...")
    tokenizer = get_tokenizer(
        cfg.vocab_size,
        model='caps_bpe',
        )
    pad_idx = tokenizer.special_tokens[cnst.TOKEN_PAD]

    print("Loading COCO dataloaders...")
    train_loader = get_coco_dataloader(
        tokenizer=tokenizer,
        batch_size=cfg.batch_size,
        max_seq_len=cfg.max_seq_len,
        resize_to=cfg.resize_to,
        datatype='train'
    )
    val_loader = get_coco_dataloader(
        tokenizer=tokenizer,
        batch_size=cfg.batch_size,
        max_seq_len=cfg.max_seq_len,
        resize_to=cfg.resize_to,
        datatype='val'
    )

    if args.lr_sweep_test:
        best_lr,res = lr_sweeping_test(
            cfg, 
            tokenizer, 
            lrs=[1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3, 1e-2],
            steps= args.lr_sweep_test_steps,
            )
        print(f'⭐  {best_lr = }')
        cfg.lr = best_lr
        # return 
    
    if args.overfit_test:
        loses = overfit_single_batch_test(
            cfg, 
            tokenizer, 
            train_loader,
            max_steps=args.overfit_test_steps,
            )
        print(f'⭐ Loss decrease:')
        for i, l in enumerate(loses) :
            if i%5 == 0:
                print(f"[Overfit] Step:{i:5} Loss: {l:9.6f}", end='  ')
                print()
        print()
        # return 

    print("👉 Building CNN-LSTM model...")
    model = CNNLSTMCaptioner(vocab_size=cfg.vocab_size, cfg=cfg, pad_idx=pad_idx)

    trainer = CNNLSTMTrainer(cfg, model, tokenizer)
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    console.print('\n[orange]' + '-'*80 + '[/orange]' )
    console.print('[bold green]\t\t👋 Hi! \t Image-Captioning (CNN-LSTM) Train/Eval Arena [/bold green]')
    console.print('[orange]' + '-'*80 + '[/orange]\n\n' )
    start = datetime.datetime.now()
    main()
    console.print('[bold green]✅ Done! Script Completed Successfuly[/bold green]')
    print(f'⌚ Script Time: {datetime.datetime.now() - start}')
