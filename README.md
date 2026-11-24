# Stanford CS231n — Deep Learning for Computer Vision

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![uv](https://img.shields.io/badge/managed%20with-uv-DE5FE9)
![Jupyter](https://img.shields.io/badge/notebooks-Jupyter-F37626?logo=jupyter&logoColor=white)

Complete solutions to all three assignments of **Stanford CS231n: Deep Learning
for Computer Vision**. Every assignment notebook was executed end-to-end and its
outputs are committed, so the reported numbers can be read directly from the
repository. All assignments share a single [uv](https://docs.astral.sh/uv/)
project and one reproducible environment.

## At a glance

| # | Assignment | Topics | Headline result |
| --- | --- | --- | --- |
| 1 | Image-classification foundations | kNN, Softmax, two-layer networks, HOG/color features, fully connected networks | **59.3%** CIFAR-10 test with HOG/color features |
| 2 | Convolutional and recurrent networks | batch/layer normalization, dropout, convolutions, PyTorch, RNN captioning | **80.5%** CIFAR-10 test; RNN overfit loss **0.0134** |
| 3 | Transformers and generative models | attention, ViT, SimCLR, DDPM, CLIP/DINO | ViT **0.4941** in 2 epochs; DINO IoU **0.643** |

All 14 assignment notebooks ran without execution errors, and every numerical
sanity check passes at float32 precision. Datasets and trained checkpoints are
not committed; each assignment's `KAGGLE.md` documents how to obtain them.

## Repository layout

```text
CS231n/
├── pyproject.toml              # one uv project for all assignments
├── uv.lock
├── .python-version
├── README.md
├── assignment1/
│   ├── README.md               # task list, implementations, results
│   ├── KAGGLE.md               # dataset and runtime notes
│   ├── cs231n/
│   ├── knn.ipynb
│   ├── softmax.ipynb
│   ├── two_layer_net.ipynb
│   ├── features.ipynb
│   └── FullyConnectedNets.ipynb
├── assignment2/
│   ├── README.md
│   ├── KAGGLE.md
│   ├── cs231n/
│   ├── BatchNormalization.ipynb
│   ├── Dropout.ipynb
│   ├── ConvolutionalNetworks.ipynb
│   ├── PyTorch.ipynb
│   └── RNN_Captioning_pytorch.ipynb
└── assignment3/
    ├── README.md
    ├── KAGGLE.md
    ├── cs231n/
    ├── Transformer_Captioning.ipynb
    ├── Self_Supervised_Learning.ipynb
    ├── DDPM.ipynb
    └── CLIP_DINO.ipynb
```

## Environment

The project targets **Python 3.11** and installs all assignment dependencies
through uv:

```bash
uv sync
uv run jupyter lab
```

The shared environment covers NumPy/SciPy, PyTorch and torchvision, Cython,
`h5py`, `imageio`, scikit-learn, pandas, `einops`, `thop`, `opencv-python`,
`openai-clip`, `ftfy` and `regex`. `setuptools<81` is pinned because
`openai-clip` still imports `pkg_resources`.

Datasets are not committed. Each notebook's setup cell looks for CIFAR-10 and
COCO under the assignment directory and under `/kaggle/input`, and links a
mounted Kaggle dataset automatically. Otherwise, run the course download
scripts from the assignment's `cs231n/datasets` directory.

## Assignment 1 — Image-classification foundations

Implements the classical pipeline from scratch: distance-based classifiers, a
Softmax classifier with analytic gradients, a modular two-layer network, image
feature representations, and fully connected networks with an SGD/Momentum/
Adam optimizer library.

| Question | Model | Target | Result |
| --- | --- | --- | --- |
| Q1 | k-nearest neighbors | ~28% test | **28.2% test** |
| Q2 | Softmax on raw pixels | >36% validation | **37.4% val / 36.3% test** |
| Q3 | Two-layer network | >48% validation/test | **52.1% val / 48.2% test** |
| Q4 | Softmax on HOG/color features | >42% validation | **43.6% val / 42.0% test** |
| Q4 | Two-layer network on HOG/color features | >55% test | **62.5% val / 59.3% test** |
| Q5 | Fully connected network | ≥50% validation/test | **52.5% val / 52.0% test** |

The executed notebooks and trained model files are in `assignment1/`. Details:
[assignment1/README.md](assignment1/README.md).

## Assignment 2 — Convolutional and recurrent networks

Implements the layer primitives and training code directly, then moves to
PyTorch and recurrent networks:

- batch normalization, layer normalization, dropout, and spatial group
  normalization;
- naive convolution and max-pooling forward/backward passes;
- `FullyConnectedNet` with normalization and dropout options;
- `ThreeLayerConvNet` and the Cython `im2col` fast layers;
- PyTorch exercises on CIFAR-10, including an open-ended ResNet-style model;
- an RNN/LSTM image-captioning model on COCO.

| Question | Check | Result |
| --- | --- | --- |
| Q1 | Batch/layer norm gradient checks | dx, dgamma, dbeta errors ≤ 1.7e-9 |
| Q2 | Dropout gradient checks | dx error 5.4e-11; 35.4% val vs 31.7% without dropout |
| Q3 | Conv/pool/fast-layer checks | errors ≤ 2.2e-8; 82% small-data train accuracy |
| Q4 | Open-ended ResNet-style model (10 epochs) | **84.5% best val / 83.0% final val / 80.5% test** |
| Q5 | RNN overfitting on small COCO data | final loss **0.0134** (< 0.1) |

Details: [assignment2/README.md](assignment2/README.md).

## Assignment 3 — Transformers and generative models

The final assignment brings together modern representation learning and
generative modelling:

- multi-head attention, positional encoding, encoder/decoder blocks, and a
  Vision Transformer;
- SimCLR contrastive pretraining and linear evaluation on CIFAR-10;
- DDPM forward/reverse diffusion, a conditional U-Net, and classifier-free
  guidance;
- CLIP zero-shot classification/retrieval and DINO features for one-shot video
  segmentation on DAVIS.

| Check | Target | Observed |
| --- | --- | --- |
| Transformer captioning overfit | loss < 0.05 | **0.0224** |
| ViT one-batch overfit | 1.00 top-1 | **1.000** |
| ViT on CIFAR-10 (2 epochs) | > 0.45 test accuracy | **0.4941** |
| SimCLR augmentation/loss sanity checks | ~0 error | **0 – 5.7e-8** |
| SimCLR linear evaluation | pretrained > baseline | **82.5% vs 15.3%** |
| DDPM forward/reverse checks | ≤ 1e-5 error | **0 – 8.1e-6** |
| CLIP similarity / zero-shot / retrieval | ≤ 1e-5 error; expected labels | **1.3e-6; exact match** |
| DINO segmentation, first / last frame | > 0.45 / > 0.50 IoU | **0.471 / 0.555** |
| DINO segmentation, all frames | > 0.55 IoU | **0.643** |

Two recipes were decisive. The ViT clears the two-epoch target only when the
number of updates is large enough: 4×4 patches, hidden 128, FFN 256, batch 16,
lr 5e-4 and weight decay 1e-4. DINO's first test frame contains just 11–32
patches of each small object class, so a two-layer MLP over the frozen features
outperforms a linear probe. Details and dataset notes:
[assignment3/README.md](assignment3/README.md) and
[assignment3/KAGGLE.md](assignment3/KAGGLE.md).

## Reproducibility notes

- The notebooks were run on Kaggle; their outputs are committed as executed.
- Datasets (`cifar-10-batches-py`, `coco_captioning`, DAVIS, emoji data) and
  model checkpoints are ignored by Git and downloaded on demand.
- `solver.py`, `layer_utils.py`, `fast_layers.py`, `gradient_check.py`,
  `data_utils.py`, `coco_utils.py` and `vis_utils.py` are course-provided
  utilities; the assignment implementations live alongside them.
- The PDF step in `collectSubmission.sh` requires a LaTeX installation
  (`xelatex`); the code archive is produced without it.
- The course's final project is not part of this repository.

## Attribution

Starter code, notebooks and assignment materials: **Stanford CS231n course
staff**. Implementations, experiments and repository organization: the
repository author. This is an unofficial coursework repository and is not
affiliated with or endorsed by Stanford University.
