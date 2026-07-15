#!/usr/bin/env python3
"""Emit timed MP3 chunks into a watch directory for ShazamFileWatchCtrl.

Unless ``--no-clean``, deletes prior chunk audio under the output directory and
under ``runtime_chunks/`` in the repo, then streams.

Set ``--out-dir`` (default ``/tmp/robo_shazam_watch`` or ``$ROBO_SHAZAM_WATCH_DIR``)
to match ``ShazamFileWatchCtrlCfg.watch_dir``.

Examples::

    python scripts/stream_shazam_source.py --song dynamite
    python scripts/stream_shazam_source.py --song swim --segment-seconds 5 --gap 4
    python scripts/stream_shazam_source.py --song butter --out-dir /tmp/mywatch

Requires ``ffmpeg`` on PATH.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CANONICAL = {
    "dynamite": ROOT / "mp3_songs" / "BTS - Dynamite (Lyrics).mp3",
    "swim": ROOT / "mp3_songs" / "BTS - SWIM (Lyrics).mp3",
    "salsa": ROOT / "mp3_songs" / "salsa musik.mp3",
}

# Same extensions as ShazamFileWatchCtrl.supported_exts
_CHUNK_EXTS = (".mp3", ".wav", ".flac", ".m4a")


def clean_chunk_dirs(*dirs: Path) -> None:
    """Remove prior chunk audio files so a new run does not mix old clips."""
    for d in dirs:
        if not d.is_dir():
            continue
        removed = 0
        for ext in _CHUNK_EXTS:
            for f in d.glob(f"*{ext}"):
                try:
                    f.unlink()
                    removed += 1
                except OSError:
                    pass
        if removed:
            print(f"Cleared {removed} chunk(s) under {d}", flush=True)


def butter_chunk_paths() -> list[Path]:
    d = ROOT / "butter_eval" / "BTS_-_Butter_Lyrics"
    if not d.is_dir():
        raise FileNotFoundError(f"Butter chunks dir missing: {d}")
    xs = sorted(d.glob("BTS_-_Butter_Lyrics_*.mp3"))
    if not xs:
        raise FileNotFoundError(f"No Butter mp3 chunks under {d}")
    return xs


def ffmpeg_extract_segment(
    src: Path,
    dst: Path,
    *,
    start_sec: float,
    duration_sec: float,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-ss",
        str(start_sec),
        "-i",
        str(src),
        "-t",
        str(duration_sec),
        "-ac",
        "1",
        "-ar",
        "44100",
        "-codec:a",
        "libmp3lame",
        "-q:a",
        "4",
        str(dst),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{r.stderr}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--song",
        choices=("butter", "salsa", "swim", "dynamite"),
        required=True,
        help="Which library source to stream.",
    )
    p.add_argument(
        "--out-dir",
        default=os.environ.get("ROBO_SHAZAM_WATCH_DIR", "/tmp/robo_shazam_watch"),
        help="Must match pipeline Shazam watch_dir (default or ROBO_SHAZAM_WATCH_DIR).",
    )
    p.add_argument(
        "--segment-seconds",
        type=float,
        default=5.0,
        help="Chunk length when slicing full mp3 sources.",
    )
    p.add_argument(
        "--gap",
        type=float,
        default=5.0,
        help="Seconds to wait after each chunk before emitting the next.",
    )
    p.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Stop after N chunks (default: run until Ctrl+C).",
    )
    p.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not delete existing chunk audio in watch dirs before streaming.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    seg = args.segment_seconds
    gap = args.gap

    if not args.no_clean:
        clean_chunk_dirs(out_dir, ROOT / "runtime_chunks")

    if args.song == "butter":
        sources = butter_chunk_paths()
        i = 0
        for src in sources:
            if args.max_chunks is not None and i >= args.max_chunks:
                break
            dst = out_dir / f"{i:04d}.mp3"
            shutil.copy2(src, dst)
            print(f"[{i}] {src.name} -> {dst}", flush=True)
            i += 1
            time.sleep(gap)
        print("Done.", file=sys.stderr)
        return

    src = CANONICAL[args.song]
    if not src.is_file():
        sys.exit(f"Missing audio file for --song {args.song}: {src}")

    i = 0
    try:
        while args.max_chunks is None or i < args.max_chunks:
            dst = out_dir / f"{i:04d}.mp3"
            ffmpeg_extract_segment(src, dst, start_sec=i * seg, duration_sec=seg)
            print(f"[{i}] {src.name} @ {i * seg:.1f}s -> {dst}", flush=True)
            i += 1
            time.sleep(gap)
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)


if __name__ == "__main__":
    main()
