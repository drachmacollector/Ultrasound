# 01 — Environment Setup

Target machine: laptop with **RTX 4060 (8GB VRAM)**. Primary IDE: **VS Code** (or Antigravity, same extension set applies). Kaggle/Colab used only as optional burst compute later (see §5).

---

## 1. `[MANUAL]` System-level prerequisites

1. **Update NVIDIA driver** to the latest Studio/Game Ready driver that supports CUDA 12.x. Check with:
   ```
   nvidia-smi
   ```
   Confirm it reports your RTX 4060 and a driver version supporting CUDA ≥ 12.1.
2. **Do not manually install a separate system-wide CUDA toolkit** unless you plan to compile custom CUDA kernels — the PyTorch pip/conda wheel ships its own CUDA runtime. Avoid version conflicts by *not* mixing a manual CUDA toolkit install with the PyTorch wheel unless you know you need `nvcc`.
3. Install **Python 3.10 or 3.11** (avoid 3.12 until you've confirmed all packages below have wheels for it).
4. Install **Git** and configure user name/email if not already done.
5. Install **VS Code** and the following extensions:
   - Python (Microsoft)
   - Pylance
   - Jupyter
   - GitLens (optional, helpful for reviewing agent-generated diffs)
   - Your AI coding agent's extension (Claude Code / Antigravity / Cursor equivalent) if it plugs into VS Code

## 2. `[MANUAL or AGENT]` Python environment

Create an isolated environment (conda recommended for easier CUDA-toolkit alignment, venv is fine too):

```bash
conda create -n fetalplane python=3.11 -y
conda activate fetalplane
```

Install PyTorch with CUDA support — **check pytorch.org for the exact current command for your CUDA version** before running this (do not hardcode a version blindly, versions shift):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify GPU visibility:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
This **must** print `True` and your RTX 4060 before proceeding. If it doesn't, resolve this before writing any training code — do not fall back to CPU training silently.

## 3. Core dependency list

Create `requirements.txt` at the repo root with (agent should pin actual tested versions once installed, this is the starting set):

```
torch
torchvision
timm                  # for RepVGG / EfficientNet / MobileNetV3 pretrained backbones
opencv-python
numpy
pandas
scikit-learn
matplotlib
seaborn
albumentations        # augmentation pipeline, supports the synthetic ego-motion transforms
pillow
tqdm
pyyaml
tensorboard            # experiment tracking (or wandb, see below)
onnx
onnxruntime-gpu        # for stretch-goal export validation
grad-cam                # pytorch-grad-cam library, saves reimplementing Grad-CAM by hand
```

**`[MANUAL] Decision point:** choose TensorBoard (fully local, zero setup, fine for a solo project) vs. Weights & Biases (nicer dashboards, requires a free account + API key). Recommendation: **TensorBoard** — keeps everything local and avoids an external account for a project with no team collaboration need. Note this decision in the repo README once made so the agent stays consistent.

## 4. Repository scaffold (agent should create this structure)

```
fetal-plane-rt/
├── configs/                  # YAML training configs per experiment
├── data/
│   ├── raw/                  # untouched downloaded datasets — see 02_DATASETS.md
│   ├── processed/            # resized/cleaned images + manifests
│   └── splits/               # train/val/test CSVs (patient-level)
├── src/
│   ├── data/                 # Dataset classes, augmentation, manifest builders
│   ├── models/               # backbone wrappers, classification head
│   ├── train/                # training loop, loss, schedulers
│   ├── eval/                 # metrics, confusion matrix, cross-device eval
│   ├── smoothing/            # temporal smoothing logic (tier-1, tier-2)
│   ├── realtime/             # webcam/file capture, inference engine, overlay UI
│   └── utils/                # logging, seeding, config loading
├── scripts/                  # one-off CLI scripts (download helpers, split builder, etc.)
├── notebooks/                # exploratory analysis only, no production logic here
├── checkpoints/               # saved model weights (gitignored)
├── logs/                     # tensorboard logs (gitignored)
├── requirements.txt
├── README.md
└── .gitignore                # must exclude data/raw, data/processed, checkpoints/, logs/
```

`[MANUAL]` Initialize git and make the first commit with just the scaffold (empty dirs via `.gitkeep`) before any data lands in `data/raw` — keeps the repo history clean and avoids accidentally committing large binary data.

## 5. `[MANUAL]` Optional: Kaggle/Colab burst-compute setup (do this later, only if needed)

Only set this up when we actually reach the "accuracy-ceiling comparison" experiment in `04_MODEL_TRAINING.md` (EfficientNetV2-S / Swin-Tiny runs that may be slow on 8GB VRAM at larger batch sizes). Steps when that time comes:
1. Zip and upload the processed `data/processed/` + `data/splits/` folders as a Kaggle Dataset (or Google Drive for Colab).
2. Copy the relevant training script + config into a notebook cell.
3. Train, download the resulting checkpoint (`.pt` file), and place it in the local `checkpoints/` folder.
4. Do **not** develop new code there — it's purely for running an already-written script faster/bigger, not for iterating on logic.

## 6. Sanity checklist before moving to Phase 2

- [ ] `nvidia-smi` shows the RTX 4060
- [ ] `torch.cuda.is_available()` returns `True`
- [ ] Repo scaffold created and committed
- [ ] `requirements.txt` installs cleanly in the fresh env
- [ ] TensorBoard (or W&B) launches and shows an empty dashboard (`tensorboard --logdir logs/`)
