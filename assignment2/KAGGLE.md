# Running Assignment 2 on Kaggle

The notebooks are self-contained, but they expect to be run from the
`assignment2` directory so that `import cs231n` resolves. Each notebook starts
with a setup cell that:

1. finds the assignment root,
2. changes into it,
3. adds it to `sys.path`,
4. links a dataset from `/kaggle/input` if one is mounted, and
5. falls back to the course download script if the dataset is missing.

## 1. Get the repository into Kaggle

Choose one of these options:

- **Git clone** (internet enabled):
  ```bash
  !git clone <your-repository-url> /kaggle/working/CS231n
  ```
- **Kaggle dataset**: upload the repository as a private Kaggle dataset, then
  copy it into the working directory:
  ```bash
  !cp -r /kaggle/input/<dataset-name>/CS231n /kaggle/working/CS231n
  ```
- **Notebook upload**: upload the `assignment2` folder and notebooks into
  `/kaggle/working/CS231n/assignment2`.

## 2. Provide CIFAR-10 and COCO

CIFAR-10 is used by `BatchNormalization.ipynb`, `Dropout.ipynb`,
`ConvolutionalNetworks.ipynb`, and `PyTorch.ipynb`. Either:

- enable **Internet** in the notebook settings and let the setup cell run
  `cs231n/datasets/get_datasets.sh`, or
- add a CIFAR-10 dataset to the notebook. The setup cell searches
  `/kaggle/input/**/cifar-10-batches-py` and links the first match into
  `cs231n/datasets/cifar-10-batches-py`.

If you add the dataset manually, make sure the directory contains the standard
`data_batch_1` … `data_batch_5` and `test_batch` files.

`RNN_Captioning_pytorch.ipynb` uses the COCO captioning archive. Either enable
Internet and let the setup cell download `coco_captioning.zip`, or mount a
Kaggle dataset containing `coco2014_captions.h5`; the setup cell searches for
that file under `/kaggle/input` and links its parent directory into
`cs231n/datasets/coco_captioning`.

## 3. Install missing packages

Kaggle provides NumPy, Matplotlib, PyTorch, and torchvision. The course code
also uses `future`, `six`, `imageio`, and `h5py`:

```bash
!pip install -q future six imageio h5py
```

The optional fast convolution layers need Cython:

```bash
!pip install -q Cython
```

If a package is already present, `pip` will report that it is satisfied.

## 4. Run the notebooks

Work from the assignment directory:

```python
%cd /kaggle/working/CS231n/assignment2
```

Then run the notebooks in this order:

1. `BatchNormalization.ipynb`
2. `Dropout.ipynb`
3. `ConvolutionalNetworks.ipynb`
4. `PyTorch.ipynb`
5. `RNN_Captioning_pytorch.ipynb`

Run all cells from top to bottom so that every notebook has visible outputs.
`ConvolutionalNetworks.ipynb` attempts to compile the Cython extension in the
fast-layers section. A build failure is not fatal: `cs231n/fast_layers.py`
falls back to a stride-based implementation.

`PyTorch.ipynb` trains a ResNet-style model for 10 epochs. On Kaggle, enable a
GPU accelerator before running it. The final cell evaluates the model on the
test set once; do not run it until the validation accuracy is satisfactory.

`RNN_Captioning_pytorch.ipynb` trains a captioning model on COCO. The
small-data overfitting experiment is the main training requirement; the
sampling section is for qualitative inspection.

## 5. Save the results

After a notebook finishes:

1. Use **File → Save Version** in Kaggle (or download the `.ipynb` directly).
2. Download the executed notebook.
3. Replace the corresponding notebook in the local repository.
4. Commit the notebook with its outputs.

Assignment 2 does not write trained model files to `cs231n/saved/`.

## 6. Common issues

- **`ModuleNotFoundError: No module named 'cs231n'`** — the working directory is
  not `assignment2`. Run `%cd /kaggle/working/CS231n/assignment2` and restart the
  notebook, or let the setup cell find the root.
- **`ModuleNotFoundError: No module named 'past'`** — install `future`:
  `!pip install -q future`.
- **`ModuleNotFoundError: No module named 'h5py'`** — install `h5py`:
  `!pip install -q h5py`.
- **CIFAR-10 not found** — enable Internet, or add a CIFAR-10 dataset and check
  that the folder is named `cifar-10-batches-py`.
- **COCO data not found** — enable Internet, or add a dataset containing
  `coco2014_captions.h5` and the associated feature/vocabulary files.
- **Cython build fails** — the notebook continues with the fallback
  implementation; install `Cython` and rerun the build cell to use the faster
  path.
- **`TypeError: Image data of dtype object cannot be converted to float`** — an
  image URL was unreachable, so `image_from_url` returned `None`. This only
  affects the COCO visualization cells, not the training or grading checks;
  those cells now skip missing images. Enable Internet if you want to see the
  samples.
- **CUDA out of memory in the open-ended PyTorch section** — reduce the batch
  size in the augmented `DataLoader` in `PyTorch.ipynb`.
