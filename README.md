# EEGPT — TG fork

Fork of [BINE022/EEGPT](https://github.com/BINE022/EEGPT), the official code for
*EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG
Signals* (NeurIPS 2024).

This fork backs an undergraduate thesis (Trabalho de Graduação) at ITA — Instituto
Tecnológico de Aeronáutica. It carries fixes that the upstream benchmark does not
run without, plus the code for the thesis' own experiments.

**Author:** Sávio Lima Rodrigues · **Advisor:** Profa. Sarah Negreiros de Carvalho Leite

---

## What this fork adds

### 1. Two fixes that unblock the Sleep-EDF benchmark

Neither of these is cosmetic: each one, on its own, makes the published benchmark
impossible to reproduce from a clean clone of `main`.

**`datasets/downstream/prepare_sleep.py:57`** — the preparation loop ran
`for sub in range(39,83)`, downloading only subjects 39–82. But
`downstream/finetune_EEGPT_SleepEDF.py:195` hardcodes a 64-subject list spanning
0–82, and fold 1 validates on `[0, 2, 4, 5, 6, 7]`. The validation set came out
empty and training aborted. Changed to `range(0,83)`.

**`downstream/finetune_EEGPT_SleepEDF.py:199-212`** — the loaders filtered
subjects with `extensions=[f'.s{i}' for i in set_train]`, but `prepare_sleep.py`
writes files named `s{i}_X_Y.pt`, so the real extension is always `.pt` and the
filter never matched. Replaced with `is_valid_file`, which filters on the basename
prefix.

### 2. Reproduction of the Sleep-EDF benchmark

One cross-subject fold, 40 epochs, unmodified hyperparameters, on an NVIDIA A100
40 GB (5 h 18 min, 114 320 steps). Held-out subjects `[0, 2, 4, 5, 6, 7]`.

| Metric | This fold (peak) | Epoch | Published (Table 4, 10 folds) |
|---|---|---|---|
| Balanced Accuracy | 0.7662 | 11 | 0.6917 ± 0.0069 |
| Cohen's Kappa | 0.7663 | 18 | 0.6857 ± 0.0019 |
| Accuracy | 0.8266 | 18 | — |

Last-epoch BAC was 0.7466. The peak is measured on the validation set across 40
epochs, so it is optimistic relative to a single test figure — the gap above the
published mean should be read with that in mind, and with the caveat that a single
fold gives no variance estimate.

### 3. macOS / Apple Silicon support

`requirements-mac.txt` is the upstream requirements list minus the CUDA-only
packages and the ones that fail to build on arm64. `requirements-mac.lock.txt` is a
`pip freeze` of a known-good environment. `downstream/smoke_mps.py` builds the
encoder and runs a forward pass on random tensors, to validate a fresh install
without needing a checkpoint or a dataset.

Apple Silicon is for environment validation only. A forward pass takes ~0.2 s for a
batch of 2 on an M3, which projects to roughly 28 h per epoch on Sleep-EDF —
training has to happen on a GPU.

### 4. SADT dataset downloader

`datasets/downstream/download_SADT.py` fetches the raw *Sustained-Attention Driving
Task* dataset (Cao et al., 2019; 27 subjects, 62 sessions, 32-channel EEG at 500 Hz,
19.6 GB) from figshare, with resume and per-file MD5 verification. The figshare web
UI sits behind a JavaScript challenge, but the public API and the file host do not,
so no browser is needed.

---

## What is being built here

The thesis compares three strategies for adapting the pretrained EEGPT encoder to
binary alert/drowsy classification on SADT, under a Leave-One-Subject-Out protocol
over 27 subjects.

| Strategy | Encoder treatment | Trainable params |
|---|---|---|
| Linear probing | fully frozen | ~10⁴ |
| Layer-wise unfreezing | last *N* blocks, η_enc = 1e-5 vs η_head = 5e-4, *N* ∈ {1,2,4} | 3×10⁶ – 1.2×10⁷ |
| LoRA | low-rank adapters on attention projections, *r* ∈ {4,8,16} | ~10⁵ |

The spatial filter and the classification head are held identical across the three,
so that any difference in performance is attributable to how the encoder is treated
and not to extra capacity downstream.

Primary metric is Balanced Accuracy per held-out subject. Strategies are compared
with a paired Wilcoxon signed-rank test (N = 27, p < 0.05), against a minimum
practically relevant gain of 1.5 percentage points.

**Status:** dataset acquired and characterised; preprocessing and training code in
progress.

### A note on the SADT labels

The published description of the task says sessions run 90 minutes. In the actual
recordings they run **41 to 117.7 minutes (median 70.25)**, with 10 of the 62
sessions under an hour — so any labelling scheme that slices fixed 30-minute blocks
off each end does not apply.

The recordings do carry the original event codes (`251`/`252` lane-departure onset,
`253` response onset, `254` response offset), 199–366 trials per session, so
reaction time is recoverable directly as `onset(253) − onset(251|252)`. Labels are
therefore derived from reaction time against the session's alert baseline (the
trimmed mean of the 10 % fastest trials, as the dataset's own tutorial defines it),
rather than from elapsed time.

---

## Setup

### Environment (macOS, Apple Silicon)

```bash
brew install python@3.11
/opt/homebrew/bin/python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install "setuptools<81" wheel      # lightning-utilities 0.8.0 imports pkg_resources
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
pip install -r requirements-mac.txt
```

Install `setuptools` before the requirements, and never run two `pip` processes
against the same virtualenv concurrently — a concurrent downgrade leaves orphaned
submodules from the newer version on disk, which surfaces as puzzling `ImportError`s
that `--force-reinstall` does not clear.

Verify:

```bash
cd downstream && PYTORCH_ENABLE_MPS_FALLBACK=1 ../.venv/bin/python smoke_mps.py
# expected: encoder output: (2, 120, 4, 512) ... === SMOKE OK ===
```

`Conv1dWithConstraint` calls `torch.renorm`, which MPS does not implement in torch
2.0 — hence `PYTORCH_ENABLE_MPS_FALLBACK=1`. Expect a warning, not an error. Keep
`precision=32` on MPS; fp16 there is unreliable in torch 2.0.

### Environment (Linux, CUDA)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Pretrained checkpoint

Download `eegpt_mcae_58chs_4s_large4E.ckpt` (973 MB) from the link in the
[upstream README](https://github.com/BINE022/EEGPT#pretrained-models) and place it
at `checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt`. The figshare share link requires a
browser; command-line clients get an AWS WAF challenge.

### Datasets

```bash
cd datasets/downstream
python prepare_sleep.py     # Sleep-EDF via braindecode, ~7 h, ~10 GB raw -> 4.3 GB of .pt
python download_SADT.py     # SADT raw from figshare, 19.6 GB, resumable
```

`downstream/utils.py` executes an import-time side effect that reads
`downstream/Data/BCIC_2a_0_38HZ/` and raises `FileNotFoundError` if it is missing.
On any machine without the BCIC-IV 2a GDF files, `mkdir -p downstream/Data/BCIC_2a_0_38HZ`
is enough to get past it.

### Training

Script naming follows `{linear_probe|finetune}_{MODEL}_{DATASET}.py`. Hyperparameters
are hardcoded — edit the script before running.

```bash
cd downstream && python finetune_EEGPT_SleepEDF.py
```

Note that despite the name, `finetune_EEGPT_SleepEDF.py` keeps the encoder frozen in
practice: `configure_optimizers` passes only `chan_conv`, `linear_probe1`,
`linear_probe2`, `cls_token` and `decoder` to the optimizer. Encoder parameters do
keep `requires_grad=True`, so gradients are computed for them and then discarded.

---

## Citation

The model, the pretrained weights and all of the original code are the work of the
EEGPT authors:

```bibtex
@inproceedings{wang2024eegpt,
  title     = {{EEGPT}: Pretrained Transformer for Universal and Reliable
               Representation of {EEG} Signals},
  author    = {Wang, Guangyu and Liu, Wenchao and He, Yuhong and Xu, Cong and
               Ma, Lin and Li, Haifeng},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2024}
}
```

The SADT dataset:

```bibtex
@article{cao2019sadt,
  title   = {Multi-channel {EEG} recordings during a sustained-attention driving task},
  author  = {Cao, Zehong and Chuang, Chun-Hsiang and King, Jung-Kai and Lin, Chin-Teng},
  journal = {Scientific Data},
  volume  = {6},
  number  = {1},
  pages   = {19},
  year    = {2019}
}
```

Licensed as upstream — see `LICENSE`.
