
# src

This directory contains the **core source code** for building, training, and evaluating a **Transformer-based Image Captioning** model.
The project integrates **Vision Transformers (ViT)** as image encoders and **Transformer decoders** for text generation.


### Setup

```bash
conda create --name image-captioning-my python==3.10 -y
conda activate image-captioning-my
pip install -r requirements.txt
```

### run main

```bash
python -m main
```

## train 

```bash
cd src

python main.py --mode train \
               --config ../configs/train_vit_caption.yaml \
               --save_dir ../checkpoints/vit_captioning/
```

### Evaluate an Existing Model

```bash
python main.py --mode eval \
               --config ../configs/train_vit_caption.yaml \
               --checkpoint ../checkpoints/vit_captioning/best_model.pt
```

### Generate Captions (Inference)

```bash
python main.py --mode infer \
               --checkpoint ../checkpoints/vit_captioning/best_model.pt \
               --image_path ../samples/sample1.jpg
```

Outputs a generated caption for the provided image.
You can also pass a folder of images using `--image_dir`.


### Resume Training

```bash
python main.py --mode train \
               --config ../configs/train_vit_caption.yaml \
               --checkpoint ../checkpoints/vit_captioning/last_epoch.pt \
               --resume True
```

Automatically loads model weights, optimizer, and scheduler states.



## Current Project Status

| Component                                  |             Status             | Description                                  |
| ------------------------------------------ | :----------------------------: | -------------------------------------------- |
| Vision Transformer (ViT-B/16) Encoder      |               🔜               |                                             |
| Transformer Decoder                        |               🔜               |                                             |
| Dataset Loader (COCO/Flickr30k)            |               🔜               |                                             |
| Trainer (XE + validation loop)             |               🔜               |                                             |
| Metrics (BLEU, CIDEr, etc.)                |               🔜               |                                             |
| Inference Script                           |               🔜               |                                             |
| Mixed Precision (AMP)                      |               🔜               |                                             |
| W&B/TensorBoard Logging                    |               🔜               |                                             |
| SCST (CIDEr optimization)                  |               🔜               |                                             |
| Multi-GPU Support                          |               🔜               |                                             |
| Dataset Augmentations                      |               🔜               |                                             |
| Pretrained ViT Variants (DeiT/Swin/DINOv2) |               🔜               |                                             |

---

## Example Config Snippet (`train_vit_caption.yaml`)

```yaml
model:
  encoder: vit_base_patch16
  decoder_layers: 6
  d_model: 768
  vocab_size: 2048

data:
  dataset: coco
  image_size: 224
  batch_size: 16
  num_workers: 4

training:
  epochs: 30
  lr: 1e-4
  weight_decay: 1e-2
  grad_clip: 1.0
  use_amp: True
  log_interval: 50
```
