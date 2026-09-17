"""Download the SADT raw dataset (Cao et al., 2019) from figshare.

Sustained-Attention Driving Task: 27 subjects, 62 sessions, 32-channel EEG at
500 Hz, stored as EEGLAB .set files (MATLAB 5.0 MAT-file format).

    article: https://doi.org/10.6084/m9.figshare.6427334
    total:   ~19.6 GB across 64 files

The figshare web UI sits behind an AWS WAF JavaScript challenge, but both the
public API (api.figshare.com) and the file host (ndownloader.figshare.com) are
reachable with a plain HTTP client, so no browser is needed.

Downloads resume on re-run: a file whose size and MD5 already match the figshare
metadata is skipped. Partially written files are continued with a Range request.

Usage:
    python download_SADT.py [--dest DIR] [--workers N]
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

ARTICLE_ID = 6427334
API_URL = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}/files?page_size=200"
CHUNK = 1 << 20  # 1 MiB


def list_files():
    with urllib.request.urlopen(API_URL, timeout=60) as r:
        return json.load(r)


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def fetch(meta, dest):
    path = os.path.join(dest, meta["name"])
    expected_size = meta["size"]
    expected_md5 = meta["supplied_md5"]

    have = os.path.getsize(path) if os.path.exists(path) else 0
    if have == expected_size:
        if md5_of(path) == expected_md5:
            return meta["name"], "skip (already complete)"
        have = 0  # corrupt: start over

    mode = "ab" if 0 < have < expected_size else "wb"
    if mode == "wb":
        have = 0

    req = urllib.request.Request(meta["download_url"])
    if have:
        req.add_header("Range", f"bytes={have}-")

    with urllib.request.urlopen(req, timeout=120) as r, open(path, mode) as fh:
        while True:
            block = r.read(CHUNK)
            if not block:
                break
            fh.write(block)

    got = md5_of(path)
    if got != expected_md5:
        os.remove(path)
        raise RuntimeError(f"{meta['name']}: MD5 mismatch ({got} != {expected_md5})")
    return meta["name"], f"ok ({expected_size / 1e6:.0f} MB)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sadt_raw"))
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    files = list_files()
    total = sum(f["size"] for f in files)
    print(f"{len(files)} files, {total / 1e9:.2f} GB -> {args.dest}", flush=True)

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, f, args.dest): f["name"] for f in files}
        for i, fut in enumerate(as_completed(futures), 1):
            name = futures[fut]
            try:
                name, status = fut.result()
                print(f"[{i}/{len(files)}] {name}: {status}", flush=True)
            except Exception as exc:  # noqa: BLE001 - report and continue
                failures.append(name)
                print(f"[{i}/{len(files)}] {name}: FAILED - {exc}", flush=True)

    if failures:
        print(f"\n{len(failures)} file(s) failed; re-run to retry: {failures}", flush=True)
        return 1
    print("\nAll files downloaded and verified.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
