import math
from collections import defaultdict
from torchvision import transforms
from dataclasses import dataclass
from pycocoevalcap.cider.cider import Cider
from nltk.translate.meteor_score import meteor_score
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import constants as cnst
from torchvision.utils import make_grid
from io import BytesIO
import numpy as np
from PIL import Image


@dataclass
class CNNLSTMConfig:
    vocab_size: int = 16384
    max_seq_len: int = 64
    resize_to: int = 512 
    batch_size: int = 128
    num_workers: int = 8

    cnn_name: str = "resnet50"
    cnn_train_backbone: bool = False
    cnn_out_dim: int = 2048 # for resnet50
    proj_dim: int = 1024
    embed_dim: int = 1024
    hidden_dim: int = 1024
    num_layers: int = 1
    dropout: float = 0.1

    lr: float = 2e-4
    weight_decay: float = 0.01
    epochs: int = 10
    grad_clip: float = 1.0
    amp: bool = True

    out_dir: str = "checkpoints/cnn_lstm"
    run_name: str = "run_debug"
    samples_per_epoch: int = 5
    print_every: int = 100
    sample_every: int = 250
    
    seed: int = 24004
    device: str = "cuda"


def _ngram_counts(tokens, n):
    counts = defaultdict(int)
    for i in range(len(tokens)-n+1):
        counts[tuple(tokens[i:i+n])] += 1
    return counts


def bleu_score(reference_tokens, candidate_tokens, max_n=4):
    precisions = []
    for n in range(1, max_n+1):
        cand_counts = _ngram_counts(candidate_tokens, n)
        if not cand_counts:
            precisions.append(0.0)
            continue
        max_ref_counts = defaultdict(int)
        for ref in reference_tokens:
            ref_counts = _ngram_counts(ref, n)
            for k, v in ref_counts.items():
                max_ref_counts[k] = max(max_ref_counts[k], v)
        overlap = 0
        total = 0
        for k, v in cand_counts.items():
            overlap += min(v, max_ref_counts.get(k, 0))
            total += v
        precisions.append(overlap / max(total, 1))

    c = len(candidate_tokens)
    r = min((abs(len(ref)-c), len(ref)) for ref in reference_tokens)[1]
    bp = 1.0 if c > r else math.exp(1 - r / max(c, 1))
    geom = 0.0
    if all(p > 0 for p in precisions):
        geom = math.exp(sum((1/max_n) * math.log(p) for p in precisions))
    bleu = bp * geom
    return {
        **{f"BLEU-{i+1}": precisions[i] for i in range(max_n)},
        "BLEU": bleu
    }


def try_meteor(ref_strs, hyp_str):
    return meteor_score(ref_strs, hyp_str)


def try_cider(ref_strs, hyp_str):
    try:
        cider = Cider()
        res = {"0": [hyp_str]}
        gts = {"0": ref_strs}
        score, _ = cider.compute_score(gts, res)
        return score
    except Exception:
        return None


def aggregate_scores(sample_scores):
    out = defaultdict(float)
    for s in sample_scores:
        for k, v in s.items():
            if v is not None:
                out[k] += v
    n = len(sample_scores)
    for k in list(out.keys()):
        out[k] /= max(n, 1)
    return dict(out)




class CNNLSTMTrainer:
    def __init__(self, cfg, model, tokenizer):
        self.cfg = cfg
        self.model = model.to(cfg.device)
        self.tokenizer = tokenizer

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=tokenizer.special_tokens[cnst.TOKEN_PAD],
            label_smoothing=0.1
        )

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )

        os.makedirs(cfg.out_dir, exist_ok=True)
        self.log_dir  = os.path.join(cfg.out_dir, cfg.run_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.scaler = torch.amp.GradScaler(enabled=cfg.amp)

        tb_dir = os.path.join(os.path.join(cfg.out_dir, "tensorboard"), cfg.run_name)
        os.makedirs(tb_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tb_dir)

        print(f"TensorBoard logging at: {tb_dir}")

    def train(self, train_loader, val_loader):
        print("Starting training loop...")
        best_bleu = 0.0
        global_step = 0

        for epoch in range(self.cfg.epochs):
            print(f"\n[Epoch {epoch+1}/{self.cfg.epochs}]")
            train_loss, global_step = self._train_one_epoch(train_loader, epoch, global_step)
            print(f"  Avg Train Loss: {train_loss:.4f}")

            # Log epoch-level loss
            self.writer.add_scalar("Loss/train_epoch", train_loss, epoch + 1)

            print("  → Evaluating...")
            val_scores, val_loss = self.evaluate(val_loader)
            print(f"  Validation Scores: {val_scores}")

            # Log validation metrics
            self.writer.add_scalar("Loss/val_epoch", val_loss, epoch + 1)
            self.writer.add_scalar("BLEU/val_epoch", val_scores.get("BLEU", 0.0), epoch + 1)

            bleu = val_scores.get("BLEU", 0.0)
            if bleu > best_bleu:
                best_bleu = bleu
                self._save_checkpoint(epoch, best=True)
                print(f"  ✅ Best BLEU updated: {best_bleu:.4f}")

            self._save_checkpoint(epoch)
            self._save_sample_predictions(val_loader, epoch)

        self.writer.close()
        print("🎯 Training complete. TensorBoard logs written.")

    def _train_one_epoch(self, loader, epoch, global_step):
        self.model.train()
        total_loss = 0.0
        loop = tqdm(loader, desc=f"Training Epoch {epoch+1}")

        for step, (images, captions) in enumerate(loop):
            images = images.to(self.cfg.device)
            captions = captions.to(self.cfg.device)

            with torch.amp.autocast(device_type=self.cfg.device, enabled=self.cfg.amp):
                logits = self.model(images, captions)
                loss = self.criterion(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    captions[:, 1:].reshape(-1)
                )

            self.scaler.scale(loss).backward()
            clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()
            global_step += 1

            self.writer.add_scalar("Loss/train_step", loss.item(), global_step)

            if (step + 1) % self.cfg.print_every == 0:
                loop.set_postfix(loss=loss.item())

            if (step + 1) % self.cfg.sample_every == 0:
                print(f"\n[Epoch {epoch+1}] Step {step+1}: Saving mid-epoch samples...")
                self._save_training_samples(images, captions, step, epoch, global_step)

        return total_loss / len(loader), global_step

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        sample_scores = []
        total_loss = 0.0
        criterion = self.criterion

        for images, captions in tqdm(loader, desc="Evaluating"):
            images = images.to(self.cfg.device)
            captions = captions.to(self.cfg.device)

            with torch.amp.autocast(device_type=self.cfg.device, enabled=self.cfg.amp):
                logits = self.model(images, captions)
                loss = criterion(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    captions[:, 1:].reshape(-1)
                )
                total_loss += loss.item()

            preds = self.model.generate(
                images,
                sos_id=self.tokenizer.special_tokens[cnst.TOKEN_SOS],
                eos_id=self.tokenizer.special_tokens[cnst.TOKEN_EOS],
                max_len=self.cfg.max_seq_len,
            )

            for ref, hyp in zip(captions, preds):
                ref_tokens = [ref.tolist()]
                sample_scores.append(bleu_score(ref_tokens, hyp))

        avg_loss = total_loss / len(loader)
        return aggregate_scores(sample_scores), avg_loss

    @torch.no_grad()
    def _save_training_samples(self, images, captions, step, epoch, global_step):
        self.model.eval()
        os.makedirs(os.path.join(self.log_dir, "samples", "mid_epoch"), exist_ok=True)

        preds = self.model.generate(
            images,
            sos_id=self.tokenizer.special_tokens[cnst.TOKEN_SOS],
            eos_id=self.tokenizer.special_tokens[cnst.TOKEN_EOS],
            max_len=self.cfg.max_seq_len,
        )

        n = min(self.cfg.samples_per_epoch, len(images))
        fig, axes = plt.subplots(1, n, figsize=(20, 5))
        for i in range(n):
            img = self._decode_img(images[i])
            axes[i].imshow(img)
            axes[i].axis("off")
            ref = self.tokenizer.decode(captions[i].tolist())
            # hyp = self.tokenizer.decode(preds[i])
            valid_tokens = self.tokenizer.all_tokens_set()
            safe_pred = [int(t.item()) if int(t.item()) in valid_tokens else cnst.TOKEN_UNK for t in preds[i] ]
            hyp = self.tokenizer.decode(safe_pred)

            axes[i].set_title(f"Ref: {ref[15:65]}...\nPred: {hyp[15:65]}...", fontsize=8)
        plt.tight_layout()

        save_path = os.path.join(
            self.log_dir,
            "samples",
            "mid_epoch",
            f"epoch{epoch+1}_step{step+1}.png"
        )
        plt.savefig(save_path)
        plt.close()
        # print(f"Saved mid-epoch samples → {save_path}")

        grid_img = make_grid(images[:n].cpu(), normalize=True, scale_each=True)
        self.writer.add_image(f"Samples/epoch{epoch+1}_step{step+1}", grid_img, global_step)
        self.model.train()

    @torch.no_grad()
    def _save_sample_predictions(self, val_loader, epoch):
        self.model.eval()
        os.makedirs(os.path.join(self.log_dir, "samples"), exist_ok=True)
        images, captions = next(iter(val_loader))
        images, captions = images[:self.cfg.samples_per_epoch], captions[:self.cfg.samples_per_epoch]
        images = images.to(self.cfg.device)
        captions = captions.to(self.cfg.device)

        preds = self.model.generate(
            images,
            sos_id=self.tokenizer.special_tokens[cnst.TOKEN_SOS],
            eos_id=self.tokenizer.special_tokens[cnst.TOKEN_EOS],
            max_len=self.cfg.max_seq_len,
        )

        n = self.cfg.samples_per_epoch
        fig, axes = plt.subplots(1, n, figsize=(20, 5))
        for i in range(n):
            img = self._decode_img(images[i])
            axes[i].imshow(img)
            axes[i].axis("off")
            ref = self.tokenizer.decode(captions[i].tolist())
            # hyp = self.tokenizer.decode(preds[i])
            valid_tokens = self.tokenizer.all_tokens_set()
            safe_pred = [int(t.item()) if int(t.item()) in valid_tokens else cnst.TOKEN_UNK for t in preds[i] ]
            hyp = self.tokenizer.decode(safe_pred)
            axes[i].set_title(f"Ref: {ref[15:65]}...\nPred: {hyp[15:65]}...", fontsize=8)

        plt.tight_layout()
        save_path = os.path.join(self.log_dir, "samples", f"epoch_{epoch+1}.png")
        plt.savefig(save_path)
        plt.close()

        grid_img = make_grid(images[:n].cpu(), normalize=True, scale_each=True)
        self.writer.add_image(f"Samples/epoch_{epoch+1}", grid_img, epoch + 1)
        print(f"Epoch {epoch+1} samples saved → {save_path}")

    def _decode_img(self, tensor):
        tensor = tensor.cpu().clone()
        tensor = torch.clamp((tensor * 0.5) + 0.5, 0, 1)
        return transforms.ToPILImage()(tensor)

    def _save_checkpoint(self, epoch, best=False):
        ckpt = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "cfg": self.cfg,
        }
        tag = "best" if best else f"epoch_{epoch+1}"
        torch.save(ckpt, os.path.join(self.log_dir, f"checkpoint_{tag}.pt"))
