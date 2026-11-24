# Running Assignment 3 on Kaggle

The notebooks are self-contained. Each starts with a setup cell that finds the
`assignment3` directory, changes into it, adds it to `sys.path`, and links
datasets mounted under `/kaggle/input` into the course locations. No dataset is
committed to the repository.

## 1. Get the repository into Kaggle

- **Git clone** (Internet enabled):
  ```bash
  !git clone <your-repository-url> /kaggle/working/CS231n
  ```
- **Kaggle dataset**: upload the repository as a private dataset and copy it to
  `/kaggle/working/CS231n`.
- **Notebook upload**: upload the `assignment3` folder into
  `/kaggle/working/CS231n/assignment3`.

Work from that directory inside each notebook:

```python
%cd /kaggle/working/CS231n/assignment3
```

The setup cell does this automatically, so it is usually enough to run the top
cell of a notebook.

## 2. Datasets

| Notebook | Data | Where it is expected |
| --- | --- | --- |
| `Transformer_Captioning.ipynb` | COCO captioning | `cs231n/datasets/coco_captioning/` |
| `Self_Supervised_Learning.ipynb` | CIFAR-10 and pretrained SimCLR weights | `data/`, `pretrained_model/` |
| `DDPM.ipynb` | Emoji dataset and pretrained diffusion checkpoint | `cs231n/datasets/`, `cs231n/exp/pretrained/` |
| `CLIP_DINO.ipynb` | COCO images/captions and DAVIS 2016 | `cs231n/datasets/coco_captioning/`, TFDS cache |

The setup cell links COCO and CIFAR-10 from `/kaggle/input` when it can find
`coco2014_captions.h5` or `cifar-10-batches-py`. Otherwise it runs the course
download scripts, which require Internet. `EmojiDataset` and
`Trainer.download_pretrained()` download their own files on first use, also
requiring Internet.

DAVIS is loaded by `tensorflow_datasets` (`davis/480p`) on first use; enable
Internet for that download. Expect several minutes for the first download.

## 3. Packages

The shared uv project already contains `einops`, `tqdm`, `pandas`,
`scikit-learn`, `thop`, `opencv-python`, and `openai-clip`. On Kaggle install
the missing ones at the top of a notebook:

```bash
!pip install -q einops tqdm pandas scikit-learn thop opencv-python ftfy regex
!pip install -q openai-clip
!pip install -q tensorflow-datasets
```

Some Kaggle images provide TensorFlow Datasets already; the command is harmless
if the package is present. CLIP model weights are downloaded by `clip.load` on
first use, and DINO weights by `torch.hub.load`.

## 4. Run the notebooks

Run in this order, top to bottom, so every cell has visible outputs:

1. `Transformer_Captioning.ipynb`
2. `Self_Supervised_Learning.ipynb`
3. `DDPM.ipynb`
4. `CLIP_DINO.ipynb`

Enable a **GPU** accelerator (Settings → Accelerator) for the ViT/SimCLR
sections; switch runtimes before starting a notebook and rerun it from the top.

## 5. Common issues

- **`ModuleNotFoundError: No module named 'cs231n'`** — run the setup cell and
  make sure the working directory is `assignment3`.
- **`No module named 'clip'` or a `pkg_resources` warning** — install
  `openai-clip` and pin `setuptools<81` (already pinned in the uv project).
- **COCO/CIFAR-10 not found** — enable Internet or mount the datasets; the
  setup cell searches `/kaggle/input`.
- **DINO/DAVIS download fails** — enable Internet; TFDS writes to
  `~/tensorflow_datasets`, which is not committed.
- **HTTP errors while displaying COCO images** — dead Flickr URLs only affect
  visualization; rerun with a different minibatch or ignore them.
- **`makepdf.py` cannot find PyPDF2 or LaTeX** — install `PyPDF2` and a TeX
  distribution for the inline PDF; the code zip is produced regardless.

## 6. Results and recipes

All four notebooks ran without error outputs and all Assignment 3 targets pass.
The two configurations that were hardest to get right:

- **ViT (0.4941 test accuracy, target > 0.45)**: 4×4 patches (64 tokens), hidden
  128, 4 heads, FFN 256, dropout 0, batch 16, lr 5e-4, weight decay 1e-4, plain
  `ToTensor` inputs. Batch 16 gives ~3,100 optimizer steps per epoch; in a
  two-epoch budget that matters far more than a wider model or augmentation.
  The one-batch overfit check reaches 1.00 at lr 1e-3.
- **DINO (0.471 / 0.555 / 0.643 mean IoU, targets > 0.45 / 0.50 / 0.55)**: the
  first test frame has only 11–32 patches of each small object class out of
  3,600, so a single linear layer missed them. A two-layer MLP
  (384 → 256 → classes, AdamW, lr 1e-3, weight decay 1e-4, 500 iterations)
  over the frozen DINO features fixes it. The evaluation cell prints per-class
  IoU and class counts for every test frame.

The PDF step in `collectSubmission.sh` still needs a LaTeX installation
(`xelatex`); the code zip is produced regardless.

## 7. Save the results

After each notebook finishes, download the executed `.ipynb`, replace the
corresponding file in the local repository, and commit it with its outputs.
Generated datasets, checkpoints, and videos are ignored by Git.
