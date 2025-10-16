# Implementation of Transformers

This repo is the code base for the implementations related to Transformers - for Computer Vision. I have tried out various image based transformer architectures starting with the Vaswani's Architecture of Transformer for language translation to the state-of-the-art architectures.


$$\LARGE \text{Application} : \texttt{Image Captioning}$$
---
This repository contains my end-to-end implementations of Transformer-based architectures for **Computer Vision**, focusing on the task of **Image Captioning**.

Starting from the foundational **"Attention is All You Need" (Vaswani et al., 2017)** architecture for language translation, I progressively extend the framework to visual understanding implementing and experimenting with several modern **Vision Transformer (ViT)** variants and their applications to image-text modeling.


### Learnings
* Text and Image tokenization
* Embeddings
* Attention - MultiHead, Grouped Query, (Flash) Attn
* Encoders - ViT, DeiT, PVT, SWIN, DINO 


### Goals
* Recreate and understand core **Transformer** architectures for vision tasks.
* Implement **ViT**, **DeiT**, **Swin Transformer**, **PVTv2**, and **DINOv2**.
* Apply these architectures for **image caption generation** using encoder–decoder setups.
* Experiment with **pretraining**, **fine-tuning**, and **zero-shot** evaluation.
* Compare the visual encoder–text decoder combinations for caption quality and efficiency.

### Progress

- [x] Tokenizer
- [x] Dataset Streamer (Loader)
- [x] Transformer - Decoder
- [x] Transformer - Encoder - ViT
- [x] DDP based Parallel - Trainer (with AutoResume and Logging)
- [ ] Transformer - Encoder - DeiT
- [ ] Transformer - Encoder - SWIN
- [ ] Transformer - Encoder - PVT 
- [ ] Transformer - Encoder - DINO
- [ ] Transformer - Encoder - SWINv2
- [ ] Transformer - Encoder - PVTv2 
- [ ] Transformer - Encoder - DINOv2


## Key Components

* **Encoder:** Vision Transformer backbone (ViT / DeiT / Swin / PVTv2 / DINOv2).
* **Decoder:** Transformer-based text generator for caption synthesis.
* **Training Objective:** Cross-Entropy loss (with optional CIDEr fine-tuning).
* **Datasets:** COCO Captions, Flickr30k, NoCaps (benchmark datasets).
* **Evaluation Metrics:** BLEU, METEOR, ROUGE-L, SPICE, CIDEr.


## Research References

* *Attention is All You Need* (Vaswani et al., 2017)
* *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* (Dosovitskiy et al., 2021)
* *DeiT: Data-Efficient Image Transformers* (Touvron et al., 2021)
* *Swin Transformer* (Liu et al., 2021)
* *PVTv2: Pyramid Vision Transformer* (Wang et al., 2022)
* *DINOv2: Learning Robust Visual Features without Supervision* (Oquab et al., 2023)

## Project Structure
check [./src/](./src/)

```
src/
│
├── encoder/           # Vision Transformer-based visual encoders
├── decoder/           # Transformer-based text decoders
├── datasets/          # COCO / Flickr30k dataset loaders
├── training/          # Training, utilities and evaluation scripts
├── checkpoints/       # Results and model weights
├── notebooks/         # Experiments and visualization
├── main.py            # entry point
└── README.md
```

---

$\large \textrm{Mahanth Yalla,} $

$ \small \textrm{M.Tech - Artificial Intelligence,} $

$ \textrm{Vision and AI Lab, CDS Dept.,} $

$ \normalsize \textrm{Indian Institute of Science.}$