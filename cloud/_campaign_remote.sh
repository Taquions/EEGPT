#!/usr/bin/env bash
#
# _campaign_remote.sh -- runs inside the GPU VM, launched by run_campaign.sh over
#                        SSH with `sudo setsid nohup` so it survives the session.
#
# Sets the machine up once, then loops over folds. Each fold's result is uploaded
# as soon as it finishes, and a fold whose result is already in the bucket is
# skipped -- so a preempted spot VM resumes where it stopped instead of starting
# over. One boot pays for `apt`, `pip` and the artifact download; doing a fold
# per VM would pay that overhead 25 times.
#
# NOT to be run directly. Placeholders are substituted by run_campaign.sh.

set -uo pipefail

BUCKET="@BUCKET@"
REPO="@REPO@"
COMMIT="@COMMIT@"
STRATEGY="@STRATEGY@"
FOLDS="@FOLDS@"
EXTRA_ARGS="@EXTRA_ARGS@"
DATASET="@DATASET@"

LOG=/var/log/tg-campaign.log
exec > >(tee -a "$LOG") 2>&1

echo "=== campaign start $(date -Iseconds) ==="
echo "strategy=$STRATEGY folds=$FOLDS commit=$COMMIT dataset=$DATASET"

# Any `exit 1` below is a setup failure: report it and stop paying, instead of
# leaving the watcher polling a machine that will never produce a fold.
on_exit() {
  local rc=$?
  if [ "$rc" != "0" ]; then
    gcloud storage cp "$LOG" "gs://$BUCKET/logs/campaign_$STRATEGY.log" || true
    echo "setup failed rc=$rc $(date -Iseconds)" | \
      gcloud storage cp - "gs://$BUCKET/sentinels/${STRATEGY}_DONE_FAIL.txt" || true
    shutdown -h +2 "setup failed" || true
  fi
}
trap on_exit EXIT
shutdown -c 2>/dev/null || true

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || { echo "GPU FAIL"; exit 1; }

# --- one-time setup -------------------------------------------------------

export DEBIAN_FRONTEND=noninteractive
# A freshly booted VM is running unattended-upgrades, which holds the dpkg lock.
# Without a timeout apt fails instantly, and since this script does not use
# `set -e` it used to march past that all the way to `python: command not found`.
apt-get -o DPkg::Lock::Timeout=600 update -qq
apt-get -o DPkg::Lock::Timeout=600 install -y -qq python3-venv python3-pip git \
  || { echo "SETUP FAIL: apt install"; exit 1; }

rm -rf /opt/venv
python3 -m venv /opt/venv || { echo "SETUP FAIL: venv"; exit 1; }
source /opt/venv/bin/activate
command -v python >/dev/null || { echo "SETUP FAIL: python not on PATH"; exit 1; }
pip install --quiet --upgrade pip

# Driver 580 is backward compatible with CUDA 11.8, and the cu118 wheels run on
# Ada (sm_89) through binary compatibility within compute capability 8.x even
# though sm_89 is absent from their arch list -- verified by cloud/gpu_smoke.sh.
pip install --quiet torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
pip install --quiet \
  pytorch-lightning==2.1.2 torchmetrics==0.11.4 einops==0.7.0 timm==0.9.16 \
  transformers==4.35.2 numpy==1.26.4 pandas==1.5.3 scipy==1.10.1 \
  "scikit-learn==1.4.1.post1" mne==1.4.2 braindecode==0.8.1 pyhealth==1.1.4 \
  peft==0.7.0 tqdm==4.65.0 matplotlib==3.7.1 tensorboard==2.16.2 h5py==3.8.0 PyYAML==6.0
python -c "import torch; assert torch.cuda.is_available(); print('cuda', torch.cuda.get_device_name(0))" \
  || { echo "SETUP FAIL: torch cannot see the GPU"; exit 1; }

rm -rf /opt/EEGPT
git clone --quiet "$REPO" /opt/EEGPT
cd /opt/EEGPT
git checkout --quiet "$COMMIT"
echo "code at $(git rev-parse --short HEAD)"

# downstream/utils.py reads this directory at import time and raises if missing.
mkdir -p checkpoint downstream/Data/BCIC_2a_0_38HZ datasets/downstream

gcloud storage cp "gs://$BUCKET/checkpoints/eegpt_mcae_58chs_4s_large4E.ckpt" checkpoint/
# The archive always unpacks to datasets/downstream/sadt, whatever variant it
# holds, so the training script's default data path needs no knowledge of which
# preprocessing produced it. Which one ran is recorded in the result JSON.
gcloud storage cp "gs://$BUCKET/datasets/$DATASET" /tmp/dataset.tar
tar -xf /tmp/dataset.tar -C datasets/downstream
# Count with find rather than a glob, and report the total size: a shell glob
# hides leading-dot files, and a name count alone would not notice a truncated
# extraction.
echo "$(find datasets/downstream/sadt -name '*.pt' | wc -l) session files, \
$(du -sh datasets/downstream/sadt | cut -f1) on disk"
echo "smallest: $(find datasets/downstream/sadt -name '*.pt' -printf '%s %p\n' | sort -n | head -1)"
echo "=== setup done $(date -Iseconds) ==="

# --- fold loop ------------------------------------------------------------

cd /opt/EEGPT/downstream
mkdir -p results_sadt

# EXTRA_ARGS holds one or more argument sets separated by ';'. A single set is
# an ordinary campaign; several turn this into an ablation that shares one boot,
# instead of paying six minutes of apt, pip and downloads per configuration.
FAILED=0
IFS=';' read -ra CONFIGS <<< "$EXTRA_ARGS"
[ ${#CONFIGS[@]} -eq 0 ] && CONFIGS=("")

for CONFIG in "${CONFIGS[@]}"; do
for FOLD in $FOLDS; do
  # The suffix that keeps configurations apart in the bucket is the one the
  # training script is told to use, so the name always matches the run.
  SUFFIX=$(echo "$CONFIG" | grep -o -- '--tag-suffix[= ][^ ]*' | sed 's/.*[= ]//')
  TAG=$(printf "%s%s_fold%02d" "$STRATEGY" "$SUFFIX" "$FOLD")
  if gcloud storage ls "gs://$BUCKET/results/$TAG.json" &>/dev/null; then
    echo "--- $TAG already in the bucket, skipping"
    continue
  fi

  echo "=== $TAG start $(date -Iseconds) ==="
  # shellcheck disable=SC2086
  python train_EEGPT_SADT.py --fold "$FOLD" --strategy "$STRATEGY" \
      --out results_sadt $CONFIG
  RC=$?

  if [ "$RC" = "0" ] && [ -f "results_sadt/$TAG.json" ]; then
    gcloud storage cp "results_sadt/$TAG.json" "gs://$BUCKET/results/$TAG.json"
    echo "=== $TAG ok $(date -Iseconds) ==="
  else
    FAILED=$((FAILED + 1))
    echo "=== $TAG FAILED rc=$RC $(date -Iseconds) ==="
  fi
  gcloud storage cp "$LOG" "gs://$BUCKET/logs/campaign_$STRATEGY.log" || true
done
done

# --- finish ---------------------------------------------------------------

gcloud storage cp --recursive results_sadt "gs://$BUCKET/logs/lightning_$STRATEGY/" || true
gcloud storage cp "$LOG" "gs://$BUCKET/logs/campaign_$STRATEGY.log" || true

if [ "$FAILED" = "0" ]; then
  echo "all folds ok $(date -Iseconds)" | \
    gcloud storage cp - "gs://$BUCKET/sentinels/${STRATEGY}_DONE_OK.txt"
else
  echo "$FAILED folds failed $(date -Iseconds)" | \
    gcloud storage cp - "gs://$BUCKET/sentinels/${STRATEGY}_DONE_FAIL.txt"
fi

echo "=== campaign end $(date -Iseconds), $FAILED failures ==="
shutdown -h +5 "campaign complete" || true
