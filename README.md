# Stanford CS231n — Coursework

This repository contains my solutions to **Stanford CS231n: Deep Learning for
Computer Vision**, together with the final course project. It is organized as a
single Python project so that all assignments share one reproducible
environment.

## About the course

[CS231n](https://cs231n.stanford.edu/) is Stanford's course on deep learning for
computer vision. The assignments cover the full image-classification pipeline,
from linear classifiers and k-nearest neighbors to fully connected networks and
convolutional networks.

The starter code, notebooks, and assignment materials are provided by Stanford
CS231n and remain the property of their authors. This is an unofficial
coursework/portfolio repository and is not affiliated with or endorsed by
Stanford University.

## Repository layout

```text
CS231n/
├── pyproject.toml          # one uv project for all assignments and the final project
├── uv.lock
├── .python-version
├── .gitignore
├── README.md
├── assignment1/
│   ├── README.md
│   ├── KAGGLE.md
│   ├── cs231n/
│   ├── knn.ipynb
│   ├── softmax.ipynb
│   ├── two_layer_net.ipynb
│   ├── features.ipynb
│   └── FullyConnectedNets.ipynb
└── final_project/          # added in the final part of the course
```

## Assignments

| Assignment | Topics | Status |
| --- | --- | --- |
| Assignment 1 | kNN, Softmax, two-layer network, image features, fully connected networks | Code complete; notebooks to be executed on Kaggle |
| Assignment 2 | Batch normalization, dropout, convolutional networks | Planned |
| Assignment 3 | Image captioning / detection / segmentation | Planned |
| Final project | Open-ended computer-vision project | Planned |

## Environment

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.11.

```bash
uv sync
cd assignment1
uv run jupyter lab
```

The notebook setup cells use `cs231n/datasets/cifar-10-batches-py` for CIFAR-10.
The dataset is not committed to this repository. You can either:

- run `bash cs231n/datasets/get_datasets.sh` from `assignment1/cs231n/datasets`, or
- mount a CIFAR-10 dataset in the execution environment; the setup cells will
  look for a `cifar-10-batches-py` directory under `/kaggle/input` and link it
  automatically.

For a step-by-step Kaggle workflow, see
[assignment1/KAGGLE.md](assignment1/KAGGLE.md).

## Assignment 1 results

The notebooks are executed on Kaggle and the resulting outputs are committed
back to this repository. The target accuracies from the assignment are:

| Question | Model | Target |
| --- | --- | --- |
| Q1 | k-nearest neighbors | ~28% test accuracy |
| Q2 | Softmax on raw pixels | >36% validation accuracy |
| Q3 | Two-layer neural network | >48% validation/test accuracy |
| Q4 | Softmax + two-layer network on HOG/color features | >42% validation, >55% test accuracy |
| Q5 | Fully connected network | >=50% validation/test accuracy |

Actual results will be added after the Kaggle run.

## Notes

- Assignment 1 code is complete. Assignment 2 layers (batch/layer
  normalization, dropout, convolution, pooling, group normalization) are left
  as starter stubs and are intentionally out of scope for this assignment.
- `solver.py`, `layer_utils.py`, `gradient_check.py`, `features.py`,
  `data_utils.py`, and `vis_utils.py` are course-provided utilities.
- Trained models are saved by the notebooks in `assignment1/cs231n/saved/`.

## Attribution

Starter code and assignment materials: Stanford CS231n course staff.
Solutions and repository organization: the repository author.
