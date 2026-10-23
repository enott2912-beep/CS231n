# Running Assignment 1 on Kaggle

The notebooks are self-contained, but they expect to be run from the
`assignment1` directory so that `import cs231n` resolves. Each notebook starts
with a setup cell that:

1. finds the assignment root,
2. changes into it,
3. adds it to `sys.path`,
4. links a CIFAR-10 dataset from `/kaggle/input` if one is mounted, and
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
- **Notebook upload**: upload the `assignment1` folder and notebooks into
  `/kaggle/working/CS231n/assignment1`.

## 2. Provide CIFAR-10

Either:

- enable **Internet** in the notebook settings and let the setup cell run
  `cs231n/datasets/get_datasets.sh`, or
- add a CIFAR-10 dataset to the notebook. The setup cell searches
  `/kaggle/input/**/cifar-10-batches-py` and links the first match into
  `cs231n/datasets/cifar-10-batches-py`.

If you add the dataset manually, make sure the directory contains the standard
`data_batch_1` … `data_batch_5` and `test_batch` files.

## 3. Install missing packages

Kaggle already provides NumPy, Matplotlib, and SciPy. The course code also uses
`future`, `six`, and `imageio`:

```bash
!pip install -q future six imageio
```

If a package is already present, `pip` will simply report that it is satisfied.

## 4. Run the notebooks

Work from the assignment directory:

```python
%cd /kaggle/working/CS231n/assignment1
```

Then run the notebooks in this order:

1. `knn.ipynb`
2. `softmax.ipynb`
3. `two_layer_net.ipynb`
4. `features.ipynb`
5. `FullyConnectedNets.ipynb`

`features.ipynb` is the slowest because it extracts HOG and color features for
the full CIFAR-10 training set. Run all cells from top to bottom so that every
notebook has visible outputs.

## 5. Save the results

After a notebook finishes:

1. Use **File → Save Version** in Kaggle (or download the `.ipynb` directly).
2. Download the executed notebook.
3. Replace the corresponding notebook in the local repository.
4. Commit the notebook with its outputs.

The notebooks write trained models into `cs231n/saved/`:

- `softmax.npy`
- `best_softmax.npy`
- `best_two_layer_net.npy`
- `best_softmax_features.npy`
- `best_two_layer_net_features.npy`

The model files from the recorded Kaggle run are committed in the repository.
The `.gitignore` excludes CIFAR-10 and temporary archives, but not `.npy` model
files.

## 6. Common issues

- **`ModuleNotFoundError: No module named 'cs231n'`** — the working directory is
  not `assignment1`. Run `%cd /kaggle/working/CS231n/assignment1` and restart the
  notebook, or let the setup cell find the root.
- **`ModuleNotFoundError: No module named 'past'`** — install `future`:
  `!pip install -q future`.
- **CIFAR-10 not found** — enable Internet, or add a CIFAR-10 dataset and check
  that the folder is named `cifar-10-batches-py`.
- **Kernel runs out of memory in Q4** — the feature matrices are small compared
  with the raw images, but close other notebooks before running the full
  feature extraction.
