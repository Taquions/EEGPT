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

LOG=/var/log/tg-campaign.log
exec > >(tee -a "$LOG") 2>&1

echo "=== campaign start $(date -Iseconds) ==="
echo "strategy=$STRATEGY folds=$FOLDS commit=$COMMIT"
shutdown -c 2>/dev/null || true

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || { echo "GPU FAIL"; exit 1; }

# --- one-time setup -------------------------------------------------------

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3-venv python3-pip git

rm -rf /opt/venv
python3 -m venv /opt/venv
source /opt/venv/bin/activate
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
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

rm -rf /opt/EEGPT
git clone --quiet "$REPO" /opt/EEGPT
cd /opt/EEGPT
git checkout --quiet "$COMMIT"
echo "code at $(git rev-parse --short HEAD)"

# downstream/utils.py reads this directory at import time and raises if missing.
mkdir -p checkpoint downstream/Data/BCIC_2a_0_38HZ datasets/downstream

gcloud storage cp "gs://$BUCKET/checkpoints/eegpt_mcae_58chs_4s_large4E.ckpt" checkpoint/
gcloud storage cp "gs://$BUCKET/datasets/sadt.tar" /tmp/
tar -xf /tmp/sadt.tar -C datasets/downstream
echo "$(ls datasets/downstream/sadt/*.pt | wc -l) session files ready"
echo "=== setup done $(date -Iseconds) ==="

# --- fold loop ------------------------------------------------------------

cd /opt/EEGPT/downstream
mkdir -p results_sadt

FAILED=0
for FOLD in $FOLDS; do
  TAG=$(printf "%s_fold%02d" "$STRATEGY" "$FOLD")
  if gcloud storage ls "gs://$BUCKET/results/$TAG.json" &>/dev/null; then
    echo "--- $TAG already in the bucket, skipping"
    continue
  fi

  echo "=== $TAG start $(date -Iseconds) ==="
  # shellcheck disable=SC2086
  python train_EEGPT_SADT.py --fold "$FOLD" --strategy "$STRATEGY" \
      --out results_sadt $EXTRA_ARGS
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
