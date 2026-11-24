# Assignment 3

This directory contains solutions to **Stanford CS231n Assignment 3**. It
covers Transformer language/vision models, self-supervised contrastive
learning, denoising diffusion models, and pretrained CLIP/DINO representations.

## Questions

| Topic | Notebook | Implementation |
| --- | --- | --- |
| Transformer captioning and ViT | `Transformer_Captioning.ipynb` | `cs231n/transformer_layers.py`, `cs231n/classifiers/transformer.py` |
| SimCLR self-supervised learning | `Self_Supervised_Learning.ipynb` | `cs231n/simclr/` |
| Denoising diffusion models | `DDPM.ipynb` | `cs231n/gaussian_diffusion.py`, `cs231n/unet.py` |
| CLIP and DINO | `CLIP_DINO.ipynb` | `cs231n/clip_dino.py` |

## What is implemented

### Transformer captioning and ViT

- Sinusoidal `PositionalEncoding`, masked `MultiHeadAttention`, decoder and
  encoder layers with residual connections and layer normalization.
- `PatchEmbedding`, a Vision Transformer with average pooling, and
  `CaptioningTransformer.forward` with a causal target mask.
- Inline answers for attention design, ViT data efficiency, and attention cost.

### SimCLR

- NT-Xent loss (`sim`, `simclr_loss_naive`, `sim_positive_pairs`,
  `compute_sim_matrix`, `simclr_loss_vectorized`).
- Random-resize/crop, flip, color-jitter and grayscale augmentation; one epoch
  of continued SimCLR training and linear-evaluation comparison.

### DDPM

- `q_sample`, the forward/backward conversion between `x_0`, `x_t` and noise,
  weighted MSE loss, and the reverse sampling step.
- Conditional U-Net with time/text conditioning, skip connections, and
  classifier-free guidance.

### CLIP and DINO

- Batch cosine similarity, zero-shot classification, and image retrieval with
  CLIP; a lightweight linear classifier over DINO patch features for DAVIS
  video segmentation.
- Inline answers for contrastive learning, DINO features, and PCA maps.

## Targets

| Check | Target |
| --- | --- |
| ViT on CIFAR-10 (2 epochs) | > 0.45 test accuracy |
| Transformer overfit on 50 COCO examples | final loss < 0.05 |
| DINO segmentation (DAVIS) | mean IoU > 0.45 / 0.50 on first/last test frame, > 0.55 over the video |
| DDPM / SimCLR sanity checks | relative errors well below 1e-5 |

The Kaggle executions were downloaded and checked. All four notebooks ran
without error outputs, and every implementation sanity check passed. The
measured training results are:

| Check | Target | Observed | Status |
| --- | --- | --- | --- |
| Transformer captioning overfit | loss < 0.05 | 0.0224 | Passed |
| ViT on CIFAR-10 (2 epochs) | > 0.45 test accuracy | 0.1936 | **Not met** |
| SimCLR augmentation/loss sanity checks | ~0 error | 0 – 5.7e-8 | Passed |
| SimCLR linear evaluation | pretrained > baseline | 82.5% vs 15.3% | Passed |
| DDPM forward/reverse checks | ≤ 1e-5 error | 0 – 8.1e-6 | Passed |
| CLIP similarity/zero-shot/retrieval | ≤ 1e-5 error; expected labels | 1.3e-6; exact match | Passed |
| DINO segmentation (first/last frame) | > 0.45 / > 0.50 IoU | 0.369 / 0.532 | First frame **not met** |
| DINO segmentation (all frames) | > 0.55 IoU | 0.574 | Passed |

The ViT and first-frame DINO targets have been retuned in the notebooks
(augmentation + lighter weight decay for the ViT; stronger regularization and
label smoothing for the DINO probe), but those cells need another Kaggle run.
Their stale outputs were cleared. See [KAGGLE.md](KAGGLE.md) for details.

## Attribution

The starter code, notebooks, and assignment materials are provided by Stanford
CS231n and remain the property of their authors. This is an unofficial
coursework repository and is not affiliated with or endorsed by Stanford
University.
