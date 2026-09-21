#!/usr/bin/env bash
#
# run_campaign.sh -- run a Leave-One-Subject-Out campaign on a GCP GPU VM.
#
# Splits into steps so each can be repeated on its own: the upload is slow and
# only needed once, and `run` can be re-issued after a spot preemption because
# the remote side skips folds whose results are already in the bucket.
#
#   ./run_campaign.sh upload                          # bucket + checkpoint + data
#   ./run_campaign.sh run linear 0-24                 # create VM, launch, detach
#   ./run_campaign.sh monitor linear                  # follow progress
#   ./run_campaign.sh collect linear                  # pull results down
#   ./run_campaign.sh summarise linear                # per-subject table
#   ./run_campaign.sh teardown                        # delete the VM
#
# The code is not uploaded: the VM clones the fork at a pinned commit, so what
# ran is always identifiable from the result. Commit before launching.
#
# Environment:
#   PROJECT  GCP project      (default: brendi-whatsapp-bot)
#   BUCKET   GCS bucket       (default: transformers-inhouse-eegpt)
#   GPU      l4 | a100        (default: l4)
#   SPOT     1 | 0            (default: 1)
#   EXTRA    extra args passed through to train_EEGPT_SADT.py

set -euo pipefail

PROJECT="${PROJECT:-brendi-whatsapp-bot}"
BUCKET="${BUCKET:-transformers-inhouse-eegpt}"
REGION="${REGION:-us-central1}"
REPO_URL="${REPO_URL:-https://github.com/Taquions/EEGPT.git}"
GPU="${GPU:-l4}"
SPOT="${SPOT:-1}"
EXTRA="${EXTRA:-}"
# Which prepared dataset the VM should pull. Variants differ in window length
# or in whether the intermediate trials are included.
DATASET="${DATASET:-sadt.tar}"
# A hard ceiling on how long an instance may live, whatever happens to the job.
# An idle VM that nobody notices is the most expensive failure available here:
# one silently-never-started campaign cost seven hours of A100 before anyone
# checked. The campaign itself is well under an hour.
MAX_RUN="${MAX_RUN:-4h}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VM_NAME="${VM_NAME:-eegpt-sadt-campaign}"
ZONE_FILE="$SCRIPT_DIR/.campaign_zone"

case "$GPU" in
  l4)   MACHINE="g2-standard-8";  ZONES="${ZONES:-us-central1-a us-central1-b us-central1-c}" ;;
  a100) MACHINE="a2-highgpu-1g";  ZONES="${ZONES:-us-central1-a us-central1-b us-central1-c us-central1-f}" ;;
  *) echo "GPU must be l4 or a100" >&2; exit 2 ;;
esac

IMAGE_FAMILY="common-cu129-ubuntu-2204-nvidia-580"
IMAGE_PROJECT="deeplearning-platform-release"

log() { echo "[$(date +%H:%M:%S)] $*"; }

expand_folds() {
  # "0-24" -> "0 1 ... 24"; "3,7,9" -> "3 7 9"; "15" -> "15"
  local spec="$1"
  if [[ "$spec" =~ ^([0-9]+)-([0-9]+)$ ]]; then
    seq "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" | tr '\n' ' '
  else
    echo "$spec" | tr ',' ' '
  fi
}

zone_of_vm() {
  [ -f "$ZONE_FILE" ] && cat "$ZONE_FILE" && return 0
  gcloud compute instances list --project="$PROJECT" \
    --filter="name=$VM_NAME" --format="value(zone)" 2>/dev/null | head -1
}

# --- steps ----------------------------------------------------------------

do_upload() {
  local ckpt="$REPO_ROOT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt"
  local data="$REPO_ROOT/datasets/downstream/sadt"
  [ -f "$ckpt" ] || { echo "missing checkpoint at $ckpt" >&2; exit 1; }
  [ -d "$data" ] || { echo "missing prepared data at $data" >&2; exit 1; }

  if ! gcloud storage buckets describe "gs://$BUCKET" --project="$PROJECT" &>/dev/null; then
    log "creating gs://$BUCKET in $REGION"
    gcloud storage buckets create "gs://$BUCKET" --project="$PROJECT" \
      --location="$REGION" --uniform-bucket-level-access
  fi

  if gcloud storage ls "gs://$BUCKET/checkpoints/eegpt_mcae_58chs_4s_large4E.ckpt" &>/dev/null; then
    log "checkpoint already uploaded"
  else
    log "uploading checkpoint (973 MB)"
    gcloud storage cp "$ckpt" "gs://$BUCKET/checkpoints/"
  fi

  data="${DATA_DIR:-$data}"
  log "packing $(ls "$data"/*.pt | wc -l | tr -d ' ') session files from $(basename "$data")"
  # COPYFILE_DISABLE stops macOS tar from attaching extended-attribute headers
  # that GNU tar on the VM then warns about for every single member.
  # Packed under the name sadt/ whatever the source directory is called, so the
  # training script's default path works for every variant.
  # -s is bsdtar's rename flag; GNU tar spells it --transform. macOS ships
  # bsdtar, and the VM only ever sees the result.
  COPYFILE_DISABLE=1 tar -cf "$SCRIPT_DIR/$DATASET" \
    -C "$REPO_ROOT/datasets/downstream" -s "|^$(basename "$data")|sadt|" \
    "$(basename "$data")"
  log "uploading $DATASET ($(du -h "$SCRIPT_DIR/$DATASET" | cut -f1))"
  # A composite upload can fail halfway ("Temporary components were not uploaded
  # correctly") and still leave the caller believing it worked. Keep the tar on
  # failure so a retry does not have to repack two gigabytes, and verify the
  # object is really there before saying the upload is done.
  if ! gcloud storage cp "$SCRIPT_DIR/$DATASET" "gs://$BUCKET/datasets/"; then
    echo "upload of $DATASET failed; the local tar is kept at $SCRIPT_DIR/$DATASET" >&2
    exit 1
  fi
  gcloud storage ls "gs://$BUCKET/datasets/$DATASET" >/dev/null || {
    echo "upload reported success but gs://$BUCKET/datasets/$DATASET is absent" >&2
    exit 1
  }
  rm -f "$SCRIPT_DIR/$DATASET"
  log "upload done"
}

do_run() {
  local strategy="${1:?strategy required}" spec="${2:?folds required}"
  local folds; folds="$(expand_folds "$spec")"

  local commit; commit="$(git -C "$REPO_ROOT" rev-parse HEAD)"
  if ! git -C "$REPO_ROOT" diff --quiet || ! git -C "$REPO_ROOT" diff --cached --quiet; then
    echo "working tree is dirty -- commit and push first, or the VM will run" >&2
    echo "a different version of the code than you are looking at." >&2
    exit 1
  fi
  if ! git -C "$REPO_ROOT" branch -r --contains "$commit" &>/dev/null; then
    echo "commit $commit is not on any remote -- push first" >&2
    exit 1
  fi
  log "strategy=$strategy folds=[$spec] commit=${commit:0:8} gpu=$GPU spot=$SPOT dataset=$DATASET"

  # Clear this strategy's sentinels: a stale one from a previous attempt makes
  # the monitor report a verdict seconds after launch, for a run that has not
  # started. Per-fold results are deliberately left in place -- they are what
  # lets a resumed campaign skip the folds it already has.
  # Two campaigns can share a strategy and differ only in --tag-suffix (an
  # attention head and a flat head are both `linear`), so the strategy alone
  # does not identify a run. Keying sentinels and logs by the suffix as well
  # keeps one campaign from reporting the other's verdict.
  local suffix; suffix="$(printf '%s' "$EXTRA" | grep -o -- '--tag-suffix[= ][^ ]*' | sed 's/.*[= ]//')"
  local runid="${strategy}${suffix}"
  gcloud storage rm "gs://$BUCKET/sentinels/${runid}_DONE_OK.txt" \
    "gs://$BUCKET/sentinels/${runid}_DONE_FAIL.txt" 2>/dev/null || true

  # Unique per launch: concurrent campaigns rendering into one fixed filename
  # raced, and the first to finish deleted the file the others were about to
  # copy ("stat local .campaign_remote.rendered.sh: No such file or directory"),
  # leaving their VMs running with nothing to do.
  local remote; remote="$(mktemp "$SCRIPT_DIR/.campaign_remote.XXXXXX.sh")"
  sed -e "s|@BUCKET@|$BUCKET|g" -e "s|@REPO@|$REPO_URL|g" \
      -e "s|@COMMIT@|$commit|g" -e "s|@STRATEGY@|$strategy|g" \
      -e "s|@FOLDS@|$folds|g" -e "s|@EXTRA_ARGS@|$EXTRA|g" \
      -e "s|@DATASET@|$DATASET|g" -e "s|@RUNID@|$runid|g" \
      "$SCRIPT_DIR/_campaign_remote.sh" > "$remote"

  # Expanded with the ${a[@]+...} guard below: under `set -u`, bash 3.2 (which is
  # what macOS ships) treats an empty array expansion as an unbound variable.
  local spot_args=()
  [ "$SPOT" = "1" ] && spot_args=(--provisioning-model=SPOT)

  local zone=""
  local existing; existing="$(zone_of_vm)"
  if [ -n "$existing" ]; then
    local state
    state=$(gcloud compute instances describe "$VM_NAME" --project="$PROJECT" \
      --zone="$existing" --format="value(status)" 2>/dev/null || echo gone)
    case "$state" in
      RUNNING)
        # A VM that has just finished a campaign reports RUNNING while its
        # `shutdown -h +5` is still pending, so reusing it blindly means racing
        # a machine that is on its way down. Cancel the shutdown first, and if
        # that does not work, treat it as unusable.
        if gcloud compute ssh "$VM_NAME" --project="$PROJECT" --zone="$existing" \
             --tunnel-through-iap --quiet --command="sudo shutdown -c" &>/dev/null; then
          log "reusing the VM already up in $existing (pending shutdown cancelled)"
          zone="$existing"
        else
          log "the VM in $existing is up but not answering -- replacing it"
          gcloud compute instances delete "$VM_NAME" --project="$PROJECT" \
            --zone="$existing" --quiet --delete-disks=all || true
          rm -f "$ZONE_FILE"
        fi ;;
      gone)
        rm -f "$ZONE_FILE" ;;
      *)
        # A TERMINATED instance still holds the name, so creating fails and
        # reusing hangs in the SSH wait. It is this script's own VM and it is
        # dead, so replace it.
        log "an earlier VM is $state in $existing -- deleting it before relaunching"
        gcloud compute instances delete "$VM_NAME" --project="$PROJECT" \
          --zone="$existing" --quiet --delete-disks=all || true
        rm -f "$ZONE_FILE" ;;
    esac
  fi
  for Z in $ZONES; do
    [ -n "$zone" ] && break
    log "trying zone $Z"
    if gcloud compute instances create "$VM_NAME" \
        --project="$PROJECT" --zone="$Z" --machine-type="$MACHINE" \
        ${spot_args[@]+"${spot_args[@]}"} --maintenance-policy=TERMINATE \
        --image-family="$IMAGE_FAMILY" --image-project="$IMAGE_PROJECT" \
        --max-run-duration="$MAX_RUN" --instance-termination-action=DELETE \
        --boot-disk-size=200 --boot-disk-type=pd-balanced \
        --metadata=install-nvidia-driver=True \
        --scopes=https://www.googleapis.com/auth/cloud-platform \
        --quiet 2>&1 | tail -2; then
      zone="$Z"; break
    fi
    log "zone $Z unavailable, trying next"
  done
  [ -n "$zone" ] || { echo "no zone could provision $MACHINE" >&2; exit 1; }
  echo "$zone" > "$ZONE_FILE"
  log "created in $zone"

  log "waiting for SSH"
  for i in $(seq 1 40); do
    gcloud compute ssh "$VM_NAME" --project="$PROJECT" --zone="$zone" \
      --tunnel-through-iap --command="true" --quiet 2>/dev/null && break
    sleep 15
    [ "$i" = 40 ] && { echo "ssh never came up" >&2; exit 1; }
  done

  # scp runs as the login user, which cannot write /opt, so stage it in the home
  # directory and move it with sudo. /opt rather than /tmp because /tmp does not
  # survive a reboot, and a preempted spot VM may come back.
  gcloud compute scp "$remote" "$VM_NAME:~/campaign.sh" \
    --project="$PROJECT" --zone="$zone" --tunnel-through-iap --quiet
  gcloud compute ssh "$VM_NAME" --project="$PROJECT" --zone="$zone" \
    --tunnel-through-iap --quiet \
    --command="sudo install -m 0755 ~/campaign.sh /opt/campaign.sh"
  rm -f "$remote"

  # systemd-run, not a GCE startup-script and not setsid+nohup. The startup-script
  # runner has an internal timeout and kills long jobs through the cgroup with no
  # Python traceback. setsid+nohup backgrounded inside `ssh --command` loses the
  # race against the session closing: the unit was staged, the command returned
  # "launching detached", and nothing ever ran -- the VM sat idle for seven hours.
  # A transient unit is owned by systemd from the moment it is created, and
  # `systemctl is-active` says so straight away.
  log "launching as a systemd unit"
  gcloud compute ssh "$VM_NAME" --project="$PROJECT" --zone="$zone" \
    --tunnel-through-iap --quiet \
    --command="sudo systemctl reset-failed tg-campaign 2>/dev/null; \
               sudo systemd-run --unit=tg-campaign --collect bash /opt/campaign.sh"

  log "confirming it is running"
  if ! gcloud compute ssh "$VM_NAME" --project="$PROJECT" --zone="$zone" \
       --tunnel-through-iap --quiet \
       --command="sleep 5; systemctl is-active tg-campaign" 2>/dev/null | grep -q active; then
    echo "the unit did not come up -- check: gcloud compute ssh $VM_NAME --zone=$zone" >&2
    echo "then: sudo journalctl -u tg-campaign -n 50" >&2
    exit 1
  fi

  log "running. follow with: $0 monitor $strategy"
}

do_monitor() {
  local strategy="${1:?strategy required}"
  local zone; zone="$(zone_of_vm)"
  while true; do
    # `gcloud storage ls` exits non-zero when nothing matches, which under
    # `set -e` with pipefail kills the assignment -- and nothing matches on the
    # first pass, every time.
    local n; n=$( { gcloud storage ls "gs://$BUCKET/results/${strategy}_fold*.json" 2>/dev/null || true; } | wc -l | tr -d ' ')
    local state="gone"
    [ -n "$zone" ] && state=$(gcloud compute instances describe "$VM_NAME" \
      --project="$PROJECT" --zone="$zone" --format="value(status)" 2>/dev/null || echo gone)

    if gcloud storage ls "gs://$BUCKET/sentinels/${strategy}_DONE_OK.txt" &>/dev/null; then
      log "done, $n folds"; return 0
    fi
    if gcloud storage ls "gs://$BUCKET/sentinels/${strategy}_DONE_FAIL.txt" &>/dev/null; then
      log "finished with failures, $n folds -- see gs://$BUCKET/logs/"; return 1
    fi
    if [ "$state" != "RUNNING" ] && [ "$state" != "STAGING" ]; then
      log "VM is $state with no sentinel and $n folds done."
      log "if it was preempted, re-issue: $0 run $strategy <folds>"
      return 1
    fi
    log "$n folds done, VM $state"
    sleep 120
  done
}

do_collect() {
  local strategy="${1:?strategy required}"
  local dest="$REPO_ROOT/downstream/results_sadt"
  mkdir -p "$dest"
  gcloud storage cp "gs://$BUCKET/results/${strategy}_fold*.json" "$dest/" 2>/dev/null || {
    echo "no results for $strategy in gs://$BUCKET/results/" >&2; exit 1; }
  log "collected into $dest"
}

do_summarise() {
  local strategy="${1:?strategy required}"
  python3 "$SCRIPT_DIR/summarise_folds.py" \
    --results "$REPO_ROOT/downstream/results_sadt" --strategy "$strategy"
}

do_teardown() {
  local zone; zone="$(zone_of_vm)"
  [ -n "$zone" ] || { log "no VM found"; return 0; }
  log "deleting $VM_NAME in $zone"
  gcloud compute instances delete "$VM_NAME" --project="$PROJECT" --zone="$zone" \
    --quiet --delete-disks=all
  rm -f "$ZONE_FILE"
  log "the bucket is left in place -- results and artifacts live there"
}

case "${1:-}" in
  upload)     do_upload ;;
  run)        shift; do_run "$@" ;;
  monitor)    shift; do_monitor "$@" ;;
  collect)    shift; do_collect "$@" ;;
  summarise)  shift; do_summarise "$@" ;;
  teardown)   do_teardown ;;
  *) sed -n '3,30p' "$0"; exit 2 ;;
esac
