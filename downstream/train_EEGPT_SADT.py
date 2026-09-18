"""Train one Leave-One-Subject-Out fold of the SADT alert/drowsy task.

One fold per invocation, so a campaign is a loop over folds and the result of
each is a small JSON the caller collects. That is also what makes spot VMs safe:
a preempted run resumes at the fold it lost, not at the beginning.

The three adaptation strategies compared in this work share this one script on
purpose. The spatial filter and the classification head are identical across
them, so any difference in performance is attributable to how the encoder is
treated rather than to extra capacity downstream -- and keeping them in one file
is what guarantees that, where three near-copies would drift apart.

    --strategy linear      encoder frozen; only the spatial filter and head train
    --strategy layerwise   last N encoder blocks unfrozen, discriminative LRs
    --strategy lora        low-rank adapters on the attention projections

Usage
-----
    python train_EEGPT_SADT.py --fold 0 --strategy linear
    python train_EEGPT_SADT.py --list-folds
"""

import argparse
import json
import os
import random
import sys
import time
from functools import partial

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from Modules.models.EEGPT_mcae import EEGTransformer
from Modules.Network.utils import Conv1dWithConstraint, LinearWithConstraint
from utils_eval import get_metrics

# The SADT montage, with the old 10-20 labels already mapped to 10-10 by
# prepare_SADT.py. Order must match the channel order in the .pt files.
SADT_CHANNELS = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FT7', 'FC3',
                 'FCZ', 'FC4', 'FT8', 'T7', 'C3', 'CZ', 'C4', 'T8', 'TP7',
                 'CP3', 'CPZ', 'CP4', 'TP8', 'P7', 'P3', 'PZ', 'P4', 'P8',
                 'O1', 'OZ', 'O2']

# EEGPT's full channel vocabulary. The checkpoint was pretrained over 58
# channels, so feeding the encoder only the 30 the dataset happens to carry may
# leave part of the spatial structure it learned unused. The spatial filter is a
# learned 1x1 convolution either way, so projecting 30 inputs up to the full
# layout costs one option's worth of parameters and is worth measuring.
FULL_CHANNELS = ['FP1', 'FPZ', 'FP2', 'AF7', 'AF3', 'AF4', 'AF8', 'F7', 'F5',
                 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3',
                 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3',
                 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1',
                 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1',
                 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ',
                 'PO4', 'PO6', 'PO8', 'O1', 'OZ', 'O2']

CHANNEL_SETS = {"sadt": SADT_CHANNELS, "full": FULL_CHANNELS}
N_INPUT_CHANNELS = len(SADT_CHANNELS)

PATCH_SIZE = 64
EMBED_DIM = 512
EMBED_NUM = 4
DEPTH = 8

# A subject needs both classes present to have a defined balanced accuracy, so
# one with an empty class cannot serve as a test fold at all.
MIN_WINDOWS_PER_CLASS = 1


def seed_everything(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# --- data --------------------------------------------------------------------

def load_subjects(data_dir):
    """Read every session .pt and group the windows by subject.

    Names the file in the error rather than letting the unpickling error surface
    on its own: a bad file inside a thousand-file archive is otherwise invisible,
    and `invalid load key` says nothing about which one. Leading-dot files are
    skipped because a shell glob hides them while os.listdir does not, so a
    stray one would be counted as fine by the setup log and then fail here.
    """
    per_subject = {}
    for name in sorted(os.listdir(data_dir)):
        if not name.endswith(".pt") or name.startswith("."):
            continue
        path = os.path.join(data_dir, name)
        try:
            blob = torch.load(path, map_location="cpu")
        except Exception as exc:
            raise RuntimeError(
                f"could not read {path} ({os.path.getsize(path)} bytes): {exc}") from exc
        sub = int(blob["subject"])
        X, y = blob["X"], blob["y"]
        if sub in per_subject:
            px, py = per_subject[sub]
            per_subject[sub] = (torch.cat([px, X]), torch.cat([py, y]))
        else:
            per_subject[sub] = (X, y)
    return per_subject


def usable_subjects(per_subject):
    """Subjects that can serve as a test fold, in ascending order."""
    out = []
    for sub, (_, y) in sorted(per_subject.items()):
        n_alert = int((y == 0).sum())
        n_drowsy = int((y == 1).sum())
        if min(n_alert, n_drowsy) >= MIN_WINDOWS_PER_CLASS:
            out.append(sub)
    return out


def split_loso(per_subject, folds, fold_idx, n_valid_subjects, seed):
    """Hold out one subject for test and a few more for validation.

    The validation split is by subject rather than by window. Windows from the
    same trial sequence are highly correlated, so a random split by window puts
    near-duplicates on both sides and makes validation optimistic -- which is
    exactly how the upstream Sleep-EDF preparation does it, and not a pattern to
    copy here. Hyperparameters are selected on this validation set and never on
    the test subject.
    """
    test_sub = folds[fold_idx]
    pool = [s for s in folds if s != test_sub]

    rng = np.random.RandomState(seed + fold_idx)
    valid_subs = sorted(rng.choice(pool, size=min(n_valid_subjects, len(pool) - 1),
                                   replace=False).tolist())
    train_subs = [s for s in pool if s not in valid_subs]

    def gather(subs):
        Xs = [per_subject[s][0] for s in subs]
        ys = [per_subject[s][1] for s in subs]
        return torch.cat(Xs), torch.cat(ys)

    return (gather(train_subs), gather(valid_subs), per_subject[test_sub],
            {"test": test_sub, "valid": valid_subs, "train": train_subs})


# --- model -------------------------------------------------------------------

class LitEEGPTSADT(pl.LightningModule):
    """EEGPT encoder plus a spatial filter and a two-layer probe head."""

    def __init__(self, args, n_times, class_weight, steps_per_epoch):
        super().__init__()
        self.args = args
        self.steps_per_epoch = steps_per_epoch
        self.channel_names = CHANNEL_SETS[args.encoder_channels]
        self.chans_num = len(self.channel_names)
        n_patches = n_times // PATCH_SIZE

        self.target_encoder = EEGTransformer(
            img_size=[self.chans_num, n_times],
            patch_size=PATCH_SIZE,
            embed_num=EMBED_NUM,
            embed_dim=EMBED_DIM,
            depth=DEPTH,
            num_heads=8,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))

        ckpt = torch.load(args.checkpoint, map_location="cpu")["state_dict"]
        weights = {k[len("target_encoder."):]: v for k, v in ckpt.items()
                   if k.startswith("target_encoder.")}
        missing, unexpected = self.target_encoder.load_state_dict(weights, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"checkpoint mismatch: {len(missing)} missing, "
                               f"{len(unexpected)} unexpected")

        self.chans_id = self.target_encoder.prepare_chan_ids(self.channel_names)

        self.chan_conv = Conv1dWithConstraint(N_INPUT_CHANNELS, self.chans_num, 1, max_norm=1)
        self.linear_probe1 = LinearWithConstraint(EMBED_NUM * EMBED_DIM, 16, max_norm=1)
        self.linear_probe2 = LinearWithConstraint(n_patches * 16, 2, max_norm=0.25)
        self.drop = nn.Dropout(p=0.5)

        self.apply_strategy()

        # The drowsy class is the minority by roughly three to one, and balanced
        # accuracy is the metric, so the loss is weighted rather than the data
        # resampled -- resampling would duplicate windows that already overlap.
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weight)
        self.threshold = 0.5
        self.valid_buffer = []
        self.is_sanity = True

    def apply_strategy(self):
        """Decide which encoder parameters train, and record how many."""
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        if self.args.strategy == "linear":
            pass  # encoder stays fully frozen
        elif self.args.strategy == "layerwise":
            blocks = self.target_encoder.blocks
            for block in blocks[len(blocks) - self.args.unfreeze_n:]:
                for p in block.parameters():
                    p.requires_grad = True
        elif self.args.strategy == "lora":
            # EEGTransformer keeps a ViT-style *fused* qkv projection --
            # blocks.{i}.attn.qkv is a single Linear(512 -> 1536) -- so there is
            # no q_proj/v_proj to target separately. Adapting "qkv" therefore
            # adapts queries, keys and values together. Splitting the fused
            # weight to reach Q and V alone would mean surgery on the pretrained
            # module, which buys a closer match to the LoRA paper at the cost of
            # departing from the checkpoint's own layout; adapting the fused
            # projection is what ViT-style LoRA normally does.
            from peft import LoraConfig, inject_adapter_in_model

            config = LoraConfig(
                r=self.args.lora_r,
                lora_alpha=self.args.lora_alpha,
                lora_dropout=self.args.lora_dropout,
                target_modules=["qkv"],
                bias="none")
            self.target_encoder = inject_adapter_in_model(config, self.target_encoder)
            for name, p in self.target_encoder.named_parameters():
                p.requires_grad = "lora_" in name
        else:
            raise ValueError(f"unknown strategy {self.args.strategy!r}")

    def head_parameters(self):
        return (list(self.chan_conv.parameters())
                + list(self.linear_probe1.parameters())
                + list(self.linear_probe2.parameters()))

    def encoder_trainable(self):
        return [p for p in self.target_encoder.parameters() if p.requires_grad]

    def forward(self, x):
        x = self.chan_conv(x)
        if self.args.strategy == "linear":
            self.target_encoder.eval()
            with torch.no_grad():
                z = self.target_encoder(x, self.chans_id.to(x))
        else:
            z = self.target_encoder(x, self.chans_id.to(x))
        h = self.linear_probe1(self.drop(z.flatten(2)))
        return self.linear_probe2(h.flatten(1))

    def training_step(self, batch, batch_idx):
        x, y = batch
        logit = self.forward(x)
        loss = self.loss_fn(logit, y.long())
        acc = (torch.argmax(logit, dim=-1) == y).float().mean()
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        self.log("train_acc", acc, on_epoch=True, on_step=False)
        return loss

    def on_validation_epoch_start(self):
        self.valid_buffer = []

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logit = self.forward(x)
        loss = self.loss_fn(logit, y.long())
        self.log("valid_loss", loss, on_epoch=True, on_step=False)
        prob = F.softmax(logit.float(), dim=-1)[:, 1]
        self.valid_buffer.append((y.detach().cpu(), prob.detach().cpu()))
        return loss

    def on_validation_epoch_end(self):
        if self.is_sanity:
            self.is_sanity = False
            return
        label = torch.cat([a for a, _ in self.valid_buffer]).numpy()
        score = torch.cat([b for _, b in self.valid_buffer]).numpy()
        metrics = ["accuracy", "balanced_accuracy", "roc_auc", "f1"]
        for key, value in get_metrics(score, label, metrics, True).items():
            self.log("valid_" + key, value, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        groups = [{"params": self.head_parameters(), "lr": self.args.lr_head}]
        enc = self.encoder_trainable()
        if enc:
            groups.append({"params": enc, "lr": self.args.lr_encoder})

        optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[g["lr"] for g in groups],
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.args.max_epochs,
            pct_start=0.2)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


# --- evaluation --------------------------------------------------------------

@torch.no_grad()
def collect_scores(model, loader, device):
    """Positive-class probability and label for every window in a loader."""
    model.eval().to(device)
    labels, scores = [], []
    for x, y in loader:
        prob = F.softmax(model.forward(x.to(device)).float(), dim=-1)[:, 1]
        labels.append(y.cpu())
        scores.append(prob.cpu())
    return torch.cat(labels).numpy(), torch.cat(scores).numpy()


def pick_threshold(label, score):
    """Threshold maximising balanced accuracy, chosen on validation only.

    The baseline leaves roughly eight points of balanced accuracy on the table:
    it reaches 0.70 AUROC against 0.62 BAC, meaning it ranks windows well and
    then cuts them in the wrong place. A fixed 0.5 is arbitrary for a task whose
    class balance varies from 0 to 80 % drowsy between sessions.

    This is not leakage: the validation subjects are disjoint from the test
    subject. It is deliberately chosen once, on the baseline, and then shared by
    every strategy, so no arm gets a calibration advantage over another.
    """
    order = np.argsort(score)
    candidates = np.unique(score[order])
    if len(candidates) > 512:  # keep it cheap on large validation sets
        candidates = np.quantile(candidates, np.linspace(0, 1, 512))
    best_t, best_bac = 0.5, -1.0
    pos, neg = label == 1, label == 0
    if not pos.any() or not neg.any():
        return 0.5
    for t in candidates:
        pred = score >= t
        bac = 0.5 * (pred[pos].mean() + (~pred[neg]).mean())
        if bac > best_bac:
            best_bac, best_t = bac, float(t)
    return best_t


def score_at(label, score, threshold):
    metrics = ["accuracy", "balanced_accuracy", "roc_auc", "f1", "cohen_kappa"]
    results = get_metrics(score, label, metrics, True, threshold=threshold)
    pred = (score >= threshold).astype(int)
    results["confusion_matrix"] = [
        [int(((label == i) & (pred == j)).sum()) for j in (0, 1)] for i in (0, 1)]
    return results


# --- main --------------------------------------------------------------------

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(here, "..", "datasets", "downstream", "sadt"))
    ap.add_argument("--checkpoint",
                    default=os.path.join(here, "..", "checkpoint",
                                         "eegpt_mcae_58chs_4s_large4E.ckpt"))
    ap.add_argument("--out", default=os.path.join(here, "results_sadt"))
    ap.add_argument("--strategy", default="linear",
                    choices=["linear", "layerwise", "lora"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--list-folds", action="store_true",
                    help="print the usable subjects and their fold indices, then exit")
    ap.add_argument("--unfreeze-n", type=int, default=2, help="layerwise: blocks to unfreeze")
    ap.add_argument("--lora-r", type=int, default=8, help="lora: adapter rank")
    ap.add_argument("--lora-alpha", type=int, default=16, help="lora: scaling factor")
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--encoder-channels", default="sadt", choices=["sadt", "full"],
                    help="channels presented to the encoder: the dataset's own 30, or "
                         "EEGPT's full 62-channel layout with the spatial filter "
                         "projecting up to it")
    ap.add_argument("--balance", default="loss",
                    choices=["loss", "sampler", "both", "none"],
                    help="how to handle the class imbalance")
    ap.add_argument("--no-tune-threshold", dest="tune_threshold", action="store_false",
                    help="report at a fixed 0.5 instead of the threshold that "
                         "maximises balanced accuracy on validation")
    ap.add_argument("--valid-subjects", type=int, default=4,
                    help="subjects held out of training for validation")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-epochs", type=int, default=25)
    ap.add_argument("--lr-head", type=float, default=5e-4)
    ap.add_argument("--lr-encoder", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--precision", default="16-mixed")
    ap.add_argument("--accelerator", default="auto")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag-suffix", default="",
                    help="appended to the result filename, to keep ablation runs apart")
    ap.add_argument("--fast-dev-run", action="store_true",
                    help="one train and one validation batch, to check the loop cheaply")
    args = ap.parse_args()

    seed_everything(args.seed)

    per_subject = load_subjects(args.data)
    folds = usable_subjects(per_subject)

    if args.list_folds:
        print(f"{len(folds)} usable subjects (both classes present):")
        for i, sub in enumerate(folds):
            _, y = per_subject[sub]
            print(f"  fold {i:2d} -> subject {sub:2d}  "
                  f"alert {int((y == 0).sum()):5d}  drowsy {int((y == 1).sum()):5d}")
        skipped = sorted(set(per_subject) - set(folds))
        if skipped:
            print(f"excluded (a class is empty): {skipped}")
        return 0

    if not 0 <= args.fold < len(folds):
        print(f"fold must be in [0, {len(folds)}); got {args.fold}", file=sys.stderr)
        return 2

    (Xtr, ytr), (Xva, yva), (Xte, yte), who = split_loso(
        per_subject, folds, args.fold, args.valid_subjects, args.seed)

    n_times = Xtr.shape[2]
    print(f"strategy={args.strategy} fold={args.fold} test subject={who['test']}")
    print(f"train {tuple(Xtr.shape)} ({int((ytr == 1).sum())} drowsy) | "
          f"valid {tuple(Xva.shape)} ({int((yva == 1).sum())} drowsy) | "
          f"test {tuple(Xte.shape)} ({int((yte == 1).sum())} drowsy)")
    print(f"valid subjects {who['valid']}", flush=True)

    counts = torch.bincount(ytr.long(), minlength=2).float()
    if args.balance in ("sampler", "both"):
        # Class weights only reshape the gradient; most batches still contain
        # almost no drowsy windows, which is what leaves the baseline predicting
        # alert 87 % of the time against 45 % recall on drowsy. Sampling fixes
        # the composition of the batch itself. Windows are drawn with
        # replacement, which is acceptable here because the minority class is
        # what gets repeated and the windows do not overlap in time.
        weights = (1.0 / counts.clamp(min=1))[ytr.long()]
        sampler = WeightedRandomSampler(weights.double(), len(weights), replacement=True)
        train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=args.batch_size,
                                  sampler=sampler, num_workers=args.num_workers,
                                  drop_last=True)
    else:
        train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  drop_last=True)
    valid_loader = DataLoader(TensorDataset(Xva, yva), batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(TensorDataset(Xte, yte), batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    # Applying both the sampler and the weighted loss corrects the same
    # imbalance twice and overshoots into the minority class.
    if args.balance in ("loss", "both"):
        class_weight = counts.sum() / (2 * counts.clamp(min=1))
    else:
        class_weight = torch.ones(2)
    print(f"balance={args.balance} class weights {class_weight.tolist()}")

    model = LitEEGPTSADT(args, n_times, class_weight, max(len(train_loader), 1))
    n_head = sum(p.numel() for p in model.head_parameters())
    n_enc = sum(p.numel() for p in model.encoder_trainable())
    print(f"trainable: head {n_head}, encoder {n_enc}, total {n_head + n_enc}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.strategy}{args.tag_suffix}_fold{args.fold:02d}"
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        precision=args.precision,
        max_epochs=args.max_epochs,
        callbacks=[
            EarlyStopping(monitor="valid_balanced_accuracy", mode="max",
                          patience=args.patience),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        logger=pl_loggers.CSVLogger(args.out, name=tag),
        enable_checkpointing=False,
        log_every_n_steps=10,
        fast_dev_run=args.fast_dev_run,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.fit(model, train_loader, valid_loader)
    train_seconds = time.time() - t0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    valid_label, valid_score = collect_scores(model, valid_loader, device)
    threshold = pick_threshold(valid_label, valid_score) if args.tune_threshold else 0.5
    valid_results = score_at(valid_label, valid_score, threshold)

    test_label, test_score = collect_scores(model, test_loader, device)
    results = score_at(test_label, test_score, threshold)
    # Reported alongside so the effect of calibration is separable from the
    # effect of the representation when the strategies are compared.
    results["balanced_accuracy_at_half"] = score_at(
        test_label, test_score, 0.5)["balanced_accuracy"]
    results["threshold"] = threshold
    results["valid_balanced_accuracy"] = valid_results["balanced_accuracy"]
    results["valid_roc_auc"] = valid_results["roc_auc"]
    results["n_test"] = int(len(test_label))
    results["n_test_drowsy"] = int((test_label == 1).sum())

    record = {
        "strategy": args.strategy,
        "fold": args.fold,
        "test_subject": who["test"],
        "valid_subjects": who["valid"],
        "n_train_subjects": len(who["train"]),
        "trainable_head": n_head,
        "trainable_encoder": n_enc,
        "train_seconds": round(train_seconds, 1),
        "epochs_run": int(trainer.current_epoch),
        "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
                   if torch.cuda.is_available() else None,
        "window_s": n_times / 256.0,
        "config": {k: v for k, v in vars(args).items() if k != "list_folds"},
        **results,
    }
    path = os.path.join(args.out, f"{tag}.json")
    if args.fast_dev_run:
        path = path.replace(".json", ".devrun.json")
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2)

    print(f"\nsubject {who['test']}: BAC {results['balanced_accuracy']:.4f} | "
          f"acc {results['accuracy']:.4f} | AUROC {results['roc_auc']:.4f} | "
          f"{train_seconds / 60:.1f} min")
    print(f"written to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
