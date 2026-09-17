#!/usr/bin/env bash
#
# gpu_smoke.sh -- stand up a GPU VM, check that this torch build really runs on
#                 it, measure one EEGPT training step, tear the VM down.
#
# The point is to spend about twenty cents answering a question that would
# otherwise be answered by a failed multi-hour campaign: does torch 2.0.0+cu118
# have kernels for an L4 (Ada, sm_89), and how fast is a step there compared to
# the A100 the Sleep-EDF fold ran on.
#
# The VM is deleted on any exit path, including Ctrl-C and errors.
#
# Usage:
#   ./gpu_smoke.sh [l4|a100]
#
# Environment:
#   PROJECT   GCP project id          (default: brendi-whatsapp-bot)
#   REGION    region                  (default: us-central1)
#   REPO      git repo to clone in VM (default: this fork)

set -euo pipefail

TARGET="${1:-l4}"
PROJECT="${PROJECT:-brendi-whatsapp-bot}"
REGION="${REGION:-us-central1}"
REPO="${REPO:-https://github.com/Taquions/EEGPT.git}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$TARGET" in
  l4)
    VM_NAME="eegpt-gpu-smoke-l4"
    MACHINE="g2-standard-8"
    ZONES="${ZONES:-us-central1-a us-central1-b us-central1-c}"
    ;;
  a100)
    VM_NAME="eegpt-gpu-smoke-a100"
    MACHINE="a2-highgpu-1g"
    ZONES="${ZONES:-us-central1-a us-central1-b us-central1-c us-central1-f}"
    ;;
  *)
    echo "usage: $0 [l4|a100]" >&2; exit 2 ;;
esac

# Same image family as the Sleep-EDF run, so the result transfers to that setup.
IMAGE_FAMILY="common-cu129-ubuntu-2204-nvidia-580"
IMAGE_PROJECT="deeplearning-platform-release"

log() { echo "[$(date +%H:%M:%S)] $*"; }

ZONE=""
cleanup() {
  local rc=$?
  if [ -n "$ZONE" ]; then
    log "deleting $VM_NAME in $ZONE"
    gcloud compute instances delete "$VM_NAME" --project="$PROJECT" --zone="$ZONE" \
      --quiet --delete-disks=all || \
      log "WARNING: delete failed -- check manually: gcloud compute instances list"
  fi
  exit $rc
}
trap cleanup EXIT INT TERM

log "creating $MACHINE (spot) from $IMAGE_FAMILY"
for Z in $ZONES; do
  log "trying zone $Z"
  if gcloud compute instances create "$VM_NAME" \
      --project="$PROJECT" --zone="$Z" \
      --machine-type="$MACHINE" \
      --provisioning-model=SPOT \
      --instance-termination-action=DELETE \
      --maintenance-policy=TERMINATE \
      --image-family="$IMAGE_FAMILY" \
      --image-project="$IMAGE_PROJECT" \
      --boot-disk-size=100 --boot-disk-type=pd-balanced \
      --metadata=install-nvidia-driver=True \
      --scopes=https://www.googleapis.com/auth/cloud-platform \
      --quiet 2>&1 | tail -2; then
    ZONE="$Z"
    break
  fi
  log "zone $Z unavailable (stockout or quota), trying next"
done

[ -n "$ZONE" ] || { echo "no zone could provision $MACHINE" >&2; exit 1; }
log "created in $ZONE"

log "waiting for SSH"
for i in $(seq 1 40); do
  if gcloud compute ssh "$VM_NAME" --project="$PROJECT" --zone="$ZONE" --tunnel-through-iap \
      --command="true" --quiet 2>/dev/null; then
    log "ssh up after $((i * 15))s"
    break
  fi
  sleep 15
  [ "$i" = 40 ] && { echo "ssh never came up" >&2; exit 1; }
done

log "copying test scripts"
gcloud compute scp \
  "$SCRIPT_DIR/_gpu_smoke_remote.sh" \
  "$SCRIPT_DIR/_check_arch.py" \
  "$SCRIPT_DIR/_check_kernels.py" \
  "$SCRIPT_DIR/_bench_encoder.py" \
  "$VM_NAME:/tmp/" \
  --project="$PROJECT" --zone="$ZONE" --tunnel-through-iap --quiet

log "running test (expect 8-12 min: driver, torch install, benchmark)"
gcloud compute ssh "$VM_NAME" --project="$PROJECT" --zone="$ZONE" --tunnel-through-iap --quiet \
  --command="sudo bash /tmp/_gpu_smoke_remote.sh '$REPO'" 2>&1 | tee "/tmp/gpu_smoke_${TARGET}.log"

echo
log "summary"
sed -n '/RESULT_BEGIN/,/RESULT_END/p' "/tmp/gpu_smoke_${TARGET}.log" | grep -v RESULT_ || \
  log "no summary block -- the run did not reach the benchmark, see the log above"

log "full log at /tmp/gpu_smoke_${TARGET}.log"
