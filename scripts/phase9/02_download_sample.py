"""
Phase 9 / Step 3f — Download the 300 analytical-sample images via
the ISIC archive's public API.

We do not use `isic image download --search` because that command
filters on metadata fields, not on an explicit isic_id list. The
public API exposes per-image URLs in `files.full.url`, which we GET
directly. No authentication needed for public images.

Run from project root (in venv):
    .venv/bin/python scripts/phase9/02_download_sample.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_CSV = REPO_ROOT / "artifacts" / "phase9" / "01_analytical_sample.csv"
IMG_DIR    = REPO_ROOT / "training_data" / "images"
LOG_TXT    = REPO_ROOT / "artifacts" / "phase9" / "02_download_sample.log"

API_BASE   = "https://api.isic-archive.com/api/v2/images"
TIMEOUT_S  = 30
MAX_RETRIES = 3
BACKOFF_S  = 2.0


def fetch_one(isic_id: str, out_path: Path, sess: requests.Session) -> tuple[bool, str]:
    """Fetch one image via API → S3 URL. Returns (ok, message)."""
    if out_path.exists() and out_path.stat().st_size > 0:
        return True, "cached"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            meta = sess.get(f"{API_BASE}/{isic_id}/", timeout=TIMEOUT_S).json()
            url  = meta["files"]["full"]["url"]
            r = sess.get(url, timeout=TIMEOUT_S, stream=True)
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
            return True, f"ok ({out_path.stat().st_size:,} B)"
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(BACKOFF_S * attempt)
                continue
            return False, f"FAILED after {MAX_RETRIES} attempts: {e}"


def main():
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    if not SAMPLE_CSV.exists():
        sys.exit(f"[FATAL] {SAMPLE_CSV} missing — run 01_select_sample.py first")

    df = pd.read_csv(SAMPLE_CSV)
    print(f"Loaded {len(df):,} target isic_ids from {SAMPLE_CSV.name}")
    print(f"Output dir: {IMG_DIR}")
    print()

    sess = requests.Session()
    sess.headers.update({"User-Agent": "phase9-shortcut-audit/1.0"})

    log_lines = [
        f"# Phase 9 Step 3f — analytical-sample download",
        f"target: {len(df)} images",
        f"out_dir: {IMG_DIR.relative_to(REPO_ROOT)}",
        "",
    ]

    n_ok = 0
    n_cached = 0
    n_fail = 0
    failures = []

    t0 = time.time()
    for _, row in tqdm(df.iterrows(), total=len(df), desc="download"):
        isic_id = row["isic_id"]
        out_path = IMG_DIR / f"{isic_id}.jpg"
        ok, msg = fetch_one(isic_id, out_path, sess)
        if ok:
            if msg == "cached":
                n_cached += 1
            else:
                n_ok += 1
        else:
            n_fail += 1
            failures.append((isic_id, msg))
    dt = time.time() - t0

    log_lines.append(f"## Summary")
    log_lines.append(f"  downloaded: {n_ok}")
    log_lines.append(f"  cached:     {n_cached}")
    log_lines.append(f"  failed:     {n_fail}")
    log_lines.append(f"  elapsed:    {dt:.1f} s")
    if failures:
        log_lines.append(f"\n## Failures")
        for isic_id, msg in failures[:50]:
            log_lines.append(f"  {isic_id}: {msg}")
        if len(failures) > 50:
            log_lines.append(f"  ... {len(failures) - 50} more")
    log_lines.append("")

    LOG_TXT.write_text("\n".join(log_lines) + "\n")
    print(f"\nDownloaded {n_ok} new, {n_cached} cached, {n_fail} failed "
          f"in {dt:.1f}s")
    print(f"Log: {LOG_TXT.relative_to(REPO_ROOT)}")

    if n_fail > 0:
        print(f"\n[WARN] {n_fail} download(s) failed; see log")
        sys.exit(1 if n_fail > len(df) * 0.05 else 0)  # >5% fail = hard exit


if __name__ == "__main__":
    main()
