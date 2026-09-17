#!/usr/bin/env bash
#
# _gpu_smoke_remote.sh -- runs inside the GPU VM, invoked over SSH by gpu_smoke.sh.
#
# Answers three questions, in order of how expensive they are to get wrong:
#   1. Does this torch build actually carry kernels for this GPU's architecture?
#      An L4 is Ada (sm_89), and the cu118 wheels were cut before Ada was common,
#      so they may only carry PTX for it, or nothing at all.
#   2. Do real kernels launch and produce correct results?
#   3. How long does one EEGPT training step take here? That number decides
#      whether the LOSO campaign is affordable on this GPU.
#
# Prints a machine-readable summary between RESULT_BEGIN/RESULT_END markers.

set -uo pipefail

REPO="${1:-https://github.com/Taquions/EEGPT.git}"

echo "=== gpu smoke start $(date -Iseconds) ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3-venv python3-pip git

rm -rf /opt/venv
python3 -m venv /opt/venv
source /opt/venv/bin/activate
pip install --quiet --upgrade pip

echo "=== installing torch 2.0.0+cu118 ==="
pip install --quiet torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118

echo "=== architecture and kernel check ==="
python /tmp/_check_arch.py

echo "=== kernel launch and correctness ==="
python /tmp/_check_kernels.py

echo "=== EEGPT encoder timing ==="
pip install --quiet pytorch-lightning==2.1.2 torchmetrics==0.11.4 einops==0.7.0 \
  timm==0.9.16 transformers==4.35.2 numpy==1.26.4

rm -rf /opt/EEGPT
git clone --quiet "$REPO" /opt/EEGPT
cd /opt/EEGPT
mkdir -p checkpoint downstream/Data/BCIC_2a_0_38HZ

cd /opt/EEGPT/downstream
python /tmp/_bench_encoder.py

echo "=== gpu smoke end $(date -Iseconds) ==="
