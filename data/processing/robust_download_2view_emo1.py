#!/usr/bin/env python3
"""Robust NeRSemble v2 downloader for the 2-view EMO-1 subset.

The official nersemble-data tool aborts the whole pass on a single
RemoteDisconnected; the TUM server drops connections regularly, which makes
full passes nearly impossible. This script downloads the same deterministic
file set with per-file retries + exponential backoff and size verification.
"""
import concurrent.futures as cf
import os
import sys
import time
from pathlib import Path

import requests


def _data_url() -> str:
    """NERSEMBLE_DATA_URL from env or ~/.config/nersemble_data/.env.

    The URL is covered by the NeRSemble data agreement and must never be
    committed or shared; it stays in the local env file only.
    """
    if os.environ.get("NERSEMBLE_DATA_URL"):
        return os.environ["NERSEMBLE_DATA_URL"].strip().strip('"')
    env_file = Path.home() / ".config/nersemble_data/.env"
    for line in env_file.read_text().splitlines():
        if line.strip().startswith("NERSEMBLE_DATA_URL"):
            return line.split("=", 1)[1].strip().strip(' "')
    raise RuntimeError("NERSEMBLE_DATA_URL not configured")


BASE = _data_url()
OUT = Path(os.environ.get("NERSEMBLE_OUT", "/home/coder/nersemble-data/data"))
SEQ = "EMO-1-shout+laugh"
CAMS = ["222200037", "220700191"]
WORKERS = 4
RETRIES = 8

def participant_files(pid: int):
    p = f"{pid:03d}"
    files = [
        f"{p}/calibration/camera_params.json",
        f"{p}/calibration/color_calibration.json",
    ]
    for cam in CAMS:
        files.append(f"{p}/sequences/{SEQ}/images/cam_{cam}.mp4")
        files.append(f"{p}/sequences/BACKGROUND/image_{cam}.jpg")
    return files


def fetch(rel: str) -> str:
    url = f"{BASE}/{rel}"
    target = OUT / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for attempt in range(RETRIES):
        try:
            with requests.get(url, stream=True, timeout=(10, 120)) as r:
                if r.status_code == 404:
                    return f"404  {rel}"
                r.raise_for_status()
                size = int(r.headers.get("content-length", -1))
                if target.exists() and target.stat().st_size == size:
                    return f"skip {rel}"
                tmp = target.with_suffix(target.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                if size >= 0 and tmp.stat().st_size != size:
                    raise IOError(f"size mismatch {tmp.stat().st_size} != {size}")
                tmp.rename(target)
                return f"ok   {rel}"
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(min(2 ** attempt, 30))
    return f"FAIL {rel}: {last_err}"


def main():
    # participant list straight from the server, same call the official
    # "nersemble-data list" command uses
    import ast
    import subprocess

    out = subprocess.run(
        ["/home/coder/venvs/inpaint-test/bin/nersemble-data", "list"],
        capture_output=True, text=True, check=True,
    ).stdout
    pids = ast.literal_eval(out[out.index("["): out.index("]") + 1])
    todo = []
    for pid in pids:
        for rel in participant_files(pid):
            t = OUT / rel
            if not t.exists() or t.stat().st_size == 0:
                todo.append(rel)
    print(f"{len(pids)} participants, {len(todo)} files to fetch", flush=True)
    fails = 0
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        for i, res in enumerate(ex.map(fetch, todo)):
            if res.startswith("FAIL"):
                fails += 1
                print(res, flush=True)
            if (i + 1) % 100 == 0:
                print(f"[{i + 1}/{len(todo)}]", flush=True)
    print(f"DONE fails={fails}", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
