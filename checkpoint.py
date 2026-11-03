"""
Resumable downloads and job checkpoints.

Every long-running step in this pipeline can be interrupted — dropped wifi, a
closed laptop, Ctrl-C — and none of them should start from zero afterwards.
Two mechanisms:

  * `download_file` uses HTTP Range requests to continue a partial download.
    The FNSPID news file is ~23 GB; losing that to a network blip is hours.
  * `Checkpoint` stores arbitrary JSON state next to the output, so a chunked
    pass over a huge CSV can pick up at the last saved boundary.

Both are safe to call repeatedly: a completed download is skipped, and a
checkpoint from a finished run is ignored once its output exists.
"""

import json
import os
import time

import requests


# ---------------------------------------------------------------------------
# Resumable download
# ---------------------------------------------------------------------------
def _human(n: float) -> str:
    return f"{n/1e9:.2f} GB" if n >= 1e9 else f"{n/1e6:.0f} MB"


# A 23 GB transfer over a couple of hours will lose its connection at least
# once — CDNs drop long-lived streams routinely. Since the .part file plus a
# Range request makes resuming free, the download retries itself rather than
# raising and making you rerun the command.
DOWNLOAD_MAX_ATTEMPTS = 100
DOWNLOAD_RETRY_WAITS = (5, 10, 20, 30, 60, 120)

# (connect timeout, read timeout). The read timeout is what fired mid-transfer;
# 300s is generous enough to ride out a slow patch without hanging forever.
DOWNLOAD_TIMEOUT = (30, 300)


def _download_once(url, tmp, chunk_mb, progress_every):
    """
    One attempt. Returns (bytes_on_disk, total_expected_or_0).

    Raises requests exceptions on network failure; the caller retries.
    """
    have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    headers = {"Range": f"bytes={have}-"} if have else {}

    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT, headers=headers) as r:
        if have and r.status_code == 200:
            # Server ignored Range and is sending the whole file; appending
            # would corrupt it, so start over.
            print("  server ignored Range; restarting from the beginning")
            have = 0
            mode = "wb"
        elif have and r.status_code == 206:
            mode = "ab"
        else:
            r.raise_for_status()
            mode = "wb"
            have = 0

        if r.status_code not in (200, 206):
            r.raise_for_status()

        # content-length is the REMAINING bytes on a 206, not the file size.
        remaining = int(r.headers.get("content-length", 0))
        total = have + remaining if remaining else 0

        start = time.time()
        written = have
        last = 0.0

        with open(tmp, mode) as f:
            for chunk in r.iter_content(chunk_size=chunk_mb * 1024 * 1024):
                f.write(chunk)
                written += len(chunk)

                now = time.time()
                if now - last < progress_every:
                    continue
                last = now
                rate = (written - have) / (now - start) if now > start else 0
                if total:
                    eta = (total - written) / rate if rate else 0
                    print(f"\r  {_human(written)} / {_human(total)} "
                          f"({written/total*100:5.1f}%)  {rate/1e6:5.1f} MB/s  "
                          f"ETA {eta/60:5.1f} min   ", end="", flush=True)
                else:
                    print(f"\r  {_human(written)}  {rate/1e6:5.1f} MB/s   ",
                          end="", flush=True)
        print()
        return written, total


def download_file(url: str, dest: str, chunk_mb: int = 4, progress_every: float = 5.0) -> None:
    """
    Stream a file to disk, resuming automatically across dropped connections.

    Writes to `<dest>.part` and only renames on completion, so a truncated file
    is never mistaken for a finished one. Each attempt resumes from the bytes
    already on disk via an HTTP Range request.
    """
    if os.path.exists(dest):
        print(f"Already downloaded: {dest}")
        return

    tmp = dest + ".part"
    have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    if have:
        print(f"Resuming {os.path.basename(dest)} from {_human(have)}")
    else:
        print(f"Downloading {url}\n  -> {dest}")

    for attempt in range(1, DOWNLOAD_MAX_ATTEMPTS + 1):
        try:
            written, total = _download_once(url, tmp, chunk_mb, progress_every)
        except requests.RequestException as e:
            pause = DOWNLOAD_RETRY_WAITS[min(attempt - 1, len(DOWNLOAD_RETRY_WAITS) - 1)]
            on_disk = os.path.getsize(tmp) if os.path.exists(tmp) else 0
            print(f"\n  {type(e).__name__} at {_human(on_disk)} "
                  f"(attempt {attempt}/{DOWNLOAD_MAX_ATTEMPTS}), retrying in {pause}s")
            time.sleep(pause)
            continue

        if total and written < total:
            # Stream ended early without raising, which happens when a CDN
            # closes the connection cleanly mid-transfer.
            print(f"  stream ended at {_human(written)} of {_human(total)}; resuming")
            time.sleep(5)
            continue

        os.replace(tmp, dest)
        print(f"  complete: {_human(os.path.getsize(dest))}")
        return

    raise SystemExit(
        f"Gave up after {DOWNLOAD_MAX_ATTEMPTS} attempts. "
        f"{_human(os.path.getsize(tmp) if os.path.exists(tmp) else 0)} is saved in {tmp} — "
        "re-run to continue from there."
    )


# ---------------------------------------------------------------------------
# Job checkpoints
# ---------------------------------------------------------------------------
class Checkpoint:
    """
    JSON state written atomically after each save.

    Atomic matters here: the failure mode being defended against is a laptop
    dying mid-write, which would otherwise leave a half-written JSON file that
    crashes the next run. Writing to a temp file and renaming means the
    checkpoint is either the old state or the new one, never a fragment.
    """

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def load(self) -> dict:
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            print(f"  checkpoint at {self.path} is unreadable; starting fresh")
            return {}

    def save(self, state: dict) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, self.path)

    def clear(self) -> None:
        for p in (self.path, self.path + ".tmp"):
            if os.path.exists(p):
                os.remove(p)