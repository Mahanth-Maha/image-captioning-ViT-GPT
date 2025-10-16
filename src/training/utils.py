from dotenv import load_dotenv
load_dotenv()

import torch.optim as optim
import torch.nn.functional as F

import os
import math
import yaml
import pytz

root_coco = os.getenv('DATA_DIR')
train_coco = os.getenv('TRAIN_DIR')
val_coco = os.getenv('VAL_DIR')
tokenizer_models_dir = os.getenv('TOK_DIR')

train_folder = os.path.join(root_coco, f"{train_coco}")
val_folder = os.path.join(root_coco, f"{val_coco}")
train_ann_file = os.path.join(root_coco, "annotations", f"captions_{train_coco}.json")
val_ann_file = os.path.join(root_coco, "annotations", f"captions_{val_coco}.json")


def get_fused_cross_entropy(ignore_index=-100):
    def loss_fn(logits, targets):
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)
        return F.cross_entropy(
            logits, targets,
            ignore_index=ignore_index,
            reduction='mean',
            label_smoothing=0.0,
            # fused=True
        )
    return loss_fn


def create_optimizer(model, lr=2e-4):
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if (len(param.shape) == 1 or 
                name.endswith('.bias') or 
                'norm' in name.lower() or 
                'embed' in name.lower()):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
    
    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": 0.1},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    
    optimizer = optim.AdamW(
        optimizer_grouped_parameters,
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,         
        weight_decay=0.1 ,
        fused=True,
        # foreach=True,
    )
    
    print(f"📜 AdamW optimizer: LR={lr}, β1=0.9, β2=0.95, WD=0.1, ε=1e-8")
    print(f"\tParameters with decay: {len(decay_params):,}")
    print(f"\tParameters without decay: {len(no_decay_params):,}")
    
    return optimizer


def create_scheduler(optimizer, total_steps, warmup_steps=2500, min_lr_percentage = 1.0):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            p = min_lr_percentage / 100
            return p + (1-p) * cosine_decay
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print(f"📜 Cosine scheduler: {total_steps:,} total steps, {warmup_steps:,} warmup")
    print(f"\tWarmup: {warmup_steps/total_steps*100:.1f}% | Final LR: {min_lr_percentage:.1f}% of peak")
    
    return scheduler
    

def load_config_from_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}

def merge_configs(base, override):
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            base[k] = merge_configs(base[k], v)
        else:
            base[k] = v
    return base

def overrides_to_dict(args):
    result = {}
    for key, value in vars(args).items():
        if value is None or key == "config_file":
            continue
        parts = key.split(".")
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return result


def convert2hr1(n):
    n = float(n)
    for u in ["", "K", "M", "B", "T"]:
        if abs(n) < 1000.0:
            return f"{n:,.1f} {u}"
        n /= 1000.0
    return f"{n:.1f} P"

def convert2hr2(n):
    for u in ['','K','M','B' 'T']:
        if abs(n) < 1000:
            return f"{n:.2f} {u}"
        n /= 1000.0
    return f"{n:.2f} P"

def fmt_time(total_seconds):
    total_seconds = float(total_seconds)
    neg = total_seconds < 0
    total_seconds = abs(total_seconds)
    d = int(total_seconds // 86400)
    h = int((total_seconds % 86400) // 3600)
    m = int((total_seconds % 3600) // 60)
    s = int(total_seconds % 60)
    return f"{'-' if neg else ''}{d}d {h:02d}h {m:02d}m {s:02d}s"

def fmt_dt_ist(dt):
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sept","Oct","Nov","Dec"]
    days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    ist = pytz.timezone("Asia/Kolkata")
    dt = dt.astimezone(ist)
    return f"{months[dt.month-1]} {dt.day:02d} {dt.year} {days[dt.weekday()]} {dt:%H:%M:%S}"

def get_time_str(time_in_secs):
    neg = False
    if time_in_secs <0:
        time_in_secs = abs(time_in_secs)
        neg = True
    days = int(time_in_secs // 86400)
    hours = int((time_in_secs % 86400) // 3600)
    minutes = int((time_in_secs % 3600) // 60)
    seconds = int(time_in_secs % 60)
    return f"{'-' if neg else ''} {days} Days {hours:2d} Hours {minutes:2d} Mins {seconds} Secs"
