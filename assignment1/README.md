# Assignment 1

This directory contains my solutions to **Stanford CS231n Assignment 1**. The
assignment introduces the basic image-classification pipeline and a modular
implementation of fully connected neural networks.

## Questions

| Question | Notebook | Implemented in |
| --- | --- | --- |
| Q1: k-Nearest Neighbor | `knn.ipynb` | `cs231n/classifiers/k_nearest_neighbor.py` |
| Q2: Softmax classifier | `softmax.ipynb` | `cs231n/classifiers/softmax.py`, `cs231n/classifiers/linear_classifier.py` |
| Q3: Two-layer neural network | `two_layer_net.ipynb` | `cs231n/layers.py`, `cs231n/classifiers/fc_net.py` |
| Q4: Image features | `features.ipynb` | Uses the Softmax and `TwoLayerNet` implementations above |
| Q5: Fully connected network | `FullyConnectedNets.ipynb` | `cs231n/classifiers/fc_net.py`, `cs231n/optim.py` |

## What is implemented

### Q1 — k-Nearest Neighbor

- `compute_distances_two_loops`, `compute_distances_one_loop`, and
  `compute_distances_no_loops` using the L2 distance.
- `predict_labels` with majority voting over the `k` nearest neighbors and
  deterministic tie-breaking toward the smaller label.
- 5-fold cross-validation over `k`.

### Q2 — Softmax classifier

- Analytic gradient of the naive Softmax loss.
- Fully vectorized Softmax loss and gradient.
- Mini-batch SGD, prediction, and validation-set hyperparameter tuning in
  `LinearClassifier`.
- Visualization of the learned class templates.

### Q3 — Two-layer network

- `affine_forward`, `affine_backward`, `relu_forward`, and `relu_backward`.
- Modular `softmax_loss`.
- `TwoLayerNet` initialization, forward pass, backward pass, and L2
  regularization.
- Training and hyperparameter tuning with `Solver`.

### Q4 — Image features

- HOG and HSV color-histogram features using the course-provided
  `cs231n/features.py`.
- Softmax classifier on the concatenated feature vectors.
- Two-layer neural network on the same features.

### Q5 — Fully connected network

- `FullyConnectedNet` with an arbitrary number of hidden layers.
- Forward and backward passes through repeated affine-ReLU blocks.
- SGD with momentum, RMSProp, and Adam update rules.
- Overfitting experiments on a small subset and a tuned CIFAR-10 model.

## Scope note

Assignment 1 uses only affine, ReLU, Softmax, and fully connected layers. The
batch/layer normalization, dropout, convolution, pooling, and group
normalization functions in `cs231n/layers.py` belong to later assignments and
are left as starter stubs. `FullyConnectedNet` raises `NotImplementedError` if
normalization or dropout is requested.

## Running

The notebooks are designed to run on Kaggle as well as locally. See
[KAGGLE.md](KAGGLE.md) for the Kaggle workflow.

Locally, from this directory:

```bash
uv run jupyter lab
```

The notebooks save trained models in `cs231n/saved/`:

- `softmax.npy`
- `best_softmax.npy`
- `best_two_layer_net.npy`
- `best_softmax_features.npy`
- `best_two_layer_net_features.npy`

## Dataset

CIFAR-10 is not committed. Run the course download script from
`cs231n/datasets`, or provide a `cifar-10-batches-py` directory in that
location. The notebook setup cells also look for a Kaggle-mounted copy under
`/kaggle/input`.

## Attribution

Starter code and assignment materials are from Stanford CS231n.
