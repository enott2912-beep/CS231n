# Assignment 2

This directory contains my solutions to **Stanford CS231n Assignment 2**. The
assignment covers normalization and regularization layers, convolutional
networks implemented from scratch, the PyTorch API, and recurrent networks for
image captioning.

## Questions

| Question | Notebook | Implemented in |
| --- | --- | --- |
| Q1: Batch normalization | `BatchNormalization.ipynb` | `cs231n/layers.py`, `cs231n/classifiers/fc_net.py` |
| Q2: Dropout | `Dropout.ipynb` | `cs231n/layers.py`, `cs231n/classifiers/fc_net.py` |
| Q3: Convolutional networks | `ConvolutionalNetworks.ipynb` | `cs231n/layers.py`, `cs231n/classifiers/cnn.py` |
| Q4: PyTorch | `PyTorch.ipynb` | Notebook only |
| Q5: RNN captioning | `RNN_Captioning_pytorch.ipynb` | `cs231n/rnn_layers_pytorch.py`, `cs231n/classifiers/rnn_pytorch.py` |

## What is implemented

### Q1 — Batch and layer normalization

- `batchnorm_forward` and `batchnorm_backward`, including running estimates for
  test time.
- `batchnorm_backward_alt`, the simplified single-expression backward pass.
- `layernorm_forward` and `layernorm_backward`.
- `FullyConnectedNet` with `normalization="batchnorm"` and
  `normalization="layernorm"`.

### Q2 — Dropout

- Inverted dropout forward and backward passes.
- `FullyConnectedNet` with `dropout_keep_ratio`, including the interaction
  between dropout, normalization, and the fully connected layers.
- The regularization experiment compares training and validation accuracy with
  and without dropout.

### Q3 — Convolutional networks

- Naive convolution forward and backward passes.
- Naive max-pooling forward and backward passes.
- Spatial batch normalization.
- Spatial group normalization.
- `ThreeLayerConvNet` using the fast convolution/pooling helpers.
- The notebook also exercises the optional Cython `im2col` extension; the
  provided `fast_layers.py` falls back to a stride-based implementation when
  the extension is not compiled.

### Q4 — PyTorch

- Barebones PyTorch two-layer and three-layer ConvNets.
- `nn.Module` and `nn.Sequential` implementations.
- An open-ended ResNet-18-style CIFAR-10 model with BatchNorm, random crops,
  horizontal flips, and Adam. The target is at least **70% validation
  accuracy** within 10 epochs.

### Q5 — Image captioning with RNNs

- Vanilla RNN step and sequence forward passes.
- Word embeddings, LSTM step, and LSTM sequence forward passes.
- `CaptioningRNN.loss` and `CaptioningRNN.sample`, supporting both `rnn` and
  `lstm` cell types. PyTorch autograd handles the backward pass.
- The small-data overfitting experiment targets a final loss below **0.1**.

## Results

The downloaded Kaggle executions were checked on 2026-10-30. All notebooks
parse correctly and contain no execution-error outputs. The numerical checks
for the implemented layers and models also pass with small relative errors.

| Question | Observed result | Status |
| --- | --- | --- |
| Q1: Batch/layer normalization | Forward/backward checks pass; best shown validation accuracy is 34.5% | Passed |
| Q2: Dropout | Forward/backward checks pass; best shown validation accuracy is 35.4% | Passed |
| Q3: Convolutional networks | Naive/fast layer checks pass; small-data train accuracy reaches 82% | Passed |
| Q4: Open-ended PyTorch model | 83.0% validation accuracy; 80.5% test accuracy | Passed |
| Q5: RNN captioning | Small-data final loss is 0.0134 | Passed |

The required targets are:

| Question | Target |
| --- | --- |
| Q4 open-ended PyTorch model | >=70% CIFAR-10 validation accuracy in 10 epochs |
| Q5 RNN overfit experiment | final training loss < 0.1 |

The Q4 and Q5 targets are met by the executed results above. Q1–Q3 are
validated by the assignment's gradient and consistency checks; their small
training demonstrations are not intended as final benchmark scores. The RNN
visualization section reports a few HTTP 404/410 errors for unavailable Flickr
images, but those warnings do not affect the numerical checks or training.

## Running the notebooks

Use the shared project environment from the repository root:

```bash
uv sync
cd assignment2
uv run jupyter lab
```

The setup cells are written for Kaggle and local execution. They find the
assignment directory, add it to `sys.path`, link datasets mounted under
`/kaggle/input`, and fall back to the course download scripts when internet is
available. See [KAGGLE.md](KAGGLE.md) for the Kaggle workflow.

Assignment 2 needs CIFAR-10 for the first four notebooks and the COCO captioning
archive for `RNN_Captioning_pytorch.ipynb`. The datasets are not committed.

## Attribution

The starter code, notebooks, and assignment materials are provided by Stanford
CS231n and remain the property of their authors. This is an unofficial
coursework repository and is not affiliated with or endorsed by Stanford
University.
