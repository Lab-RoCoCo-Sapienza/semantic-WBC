"""CLAP embedding fallback when acoustic fingerprint has no DB hit.

Uses a LAION CLAP checkpoint (HuggingFace) to embed the query clip and rank
precomputed embeddings of songs in ``mp3_songs/``. This is **retrieval by
similarity**, not Shazam-style temporal alignment: offset is unknown.

Requires: ``transformers``, ``torch`` (already in repo). First run downloads weights.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile

# General-audio CLAP; for music-heavy libraries you can try ``laion/larger_clap_music``.
DEFAULT_CLAP_MODEL = "laion/larger_clap_general"

# Index: 10 s from song mid; query: first 10 s (typical chunk length).
DEFAULT_EMBED_SECONDS = 10.0


@dataclass
class CLAPNearest:
    song_path: str
    similarity: float  # cosine in [-1, 1], higher = more similar
    song_id: str
    offset_sec: float | None = None  # start time of best matching window in the song, if known
    window_sec: float | None = None  # duration of the best matching window (same as build-time)


def _ffmpeg_to_wav_48k_mono(
    input_path: str,
    out_wav: str,
    *,
    ss: float | None = None,
    duration_sec: float | None = None,
) -> None:
    cmd = [
        "ffmpeg", "-y",
        *(["-ss", str(ss)] if ss is not None else []),
        *(["-t", str(duration_sec)] if duration_sec is not None else []),
        "-i", input_path,
        "-ac", "1",
        "-ar", "48000",
        "-vn",
        "-acodec", "pcm_s16le",
        out_wav,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{proc.stderr}")


def _read_wav_float_mono(path: str) -> tuple[np.ndarray, int]:
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    peak = float(np.max(np.abs(data))) if data.size else 1.0
    if peak > 0:
        data = data / peak
    return data, sr


def _duration_sec(path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)


def song_id_from_path(song_path: str) -> str:
    return hashlib.sha1(os.path.abspath(song_path).encode("utf-8")).hexdigest()[:16]


class CLAPRetriever:
    def __init__(
        self,
        model_name: str = DEFAULT_CLAP_MODEL,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self._device = device
        self._model = None
        self._processor = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import ClapModel, ClapProcessor
        except ImportError as exc:
            raise ImportError(
                "Install transformers for CLAP fallback: pip install transformers"
            ) from exc
        import torch
        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = ClapProcessor.from_pretrained(self.model_name)
        self._model = ClapModel.from_pretrained(self.model_name).to(device)
        self._model.eval()
        self._torch_device = device

    def prewarm(self) -> None:
        """Load HuggingFace weights now so the first ``embed_file`` / ``nearest`` call is fast."""
        self._lazy_load()

    def embed_file(
        self,
        audio_path: str,
        *,
        max_seconds: float = DEFAULT_EMBED_SECONDS,
        segment_from_middle: bool = False,
    ) -> np.ndarray:
        """Return L2-normalized embedding vector (1D float32)."""
        self._lazy_load()
        import torch

        if segment_from_middle:
            dur = max(0.0, _duration_sec(audio_path))
            half = max_seconds / 2.0
            ss = max(0.0, dur / 2.0 - half)
            clip_len = min(max_seconds, dur)
        else:
            ss = None
            clip_len = max_seconds

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        try:
            _ffmpeg_to_wav_48k_mono(audio_path, wav_path, ss=ss, duration_sec=clip_len)
            audio, sr = _read_wav_float_mono(wav_path)
        finally:
            if os.path.exists(wav_path):
                os.unlink(wav_path)

        if sr != 48000:
            raise ValueError(f"Expected 48000 Hz after ffmpeg, got {sr}")
        if audio.size == 0:
            raise RuntimeError(f"CLAP embed_file: decoded empty audio from {audio_path}")

        self._lazy_load()
        with torch.no_grad():
            # Newer transformers expects `audio=` (not deprecated `audios=`).
            inputs = self._processor(
                audio=audio,
                sampling_rate=48000,
                return_tensors="pt",
            )
            inputs = {k: v.to(self._torch_device) for k, v in inputs.items()}
            if hasattr(self._model, "get_audio_features"):
                emb = self._model.get_audio_features(**inputs)
            else:
                out = self._model(**inputs)
                if hasattr(out, "audio_embeds") and out.audio_embeds is not None:
                    emb = out.audio_embeds
                elif hasattr(out, "pooler_output") and out.pooler_output is not None:
                    emb = out.pooler_output
                elif isinstance(out, tuple) and len(out) > 0:
                    emb = out[0]
                else:
                    raise RuntimeError(f"Unsupported CLAP output type: {type(out)}")

            # Some transformer versions return model outputs from get_audio_features too.
            if hasattr(emb, "audio_embeds") and emb.audio_embeds is not None:
                emb = emb.audio_embeds
            elif hasattr(emb, "pooler_output") and emb.pooler_output is not None:
                emb = emb.pooler_output
            elif isinstance(emb, tuple) and len(emb) > 0:
                emb = emb[0]

            if not hasattr(emb, "detach"):
                raise RuntimeError(f"Expected tensor-like embedding, got {type(emb)}")

            vec = emb.detach().cpu().numpy().astype(np.float32).reshape(-1)
        n = np.linalg.norm(vec)
        if n > 0:
            vec = vec / n
        return vec

    def nearest(
        self,
        query_path: str,
        paths: list[str],
        matrix: np.ndarray,
        *,
        max_seconds: float = DEFAULT_EMBED_SECONDS,
        windows: list[dict] | None = None,
    ) -> CLAPNearest | None:
        """Return nearest song (and window offset when ``windows`` metadata is available).

        ``matrix`` shape (N, D), rows L2-normalized. If ``windows`` is provided, each
        row corresponds to ``windows[i] = {"path", "start_sec", "duration_sec"}`` and the
        best match reports the window start as the estimated offset in the source song.
        """
        if matrix.size == 0 or not paths:
            return None
        q = self.embed_file(query_path, max_seconds=max_seconds, segment_from_middle=False)
        sims = matrix @ q
        i = int(np.argmax(sims))
        if windows and i < len(windows):
            w = windows[i]
            song_path = w.get("path", paths[i] if i < len(paths) else "")
            offset = float(w.get("start_sec", 0.0))
            dur = float(w.get("duration_sec", max_seconds))
        else:
            song_path = paths[i]
            offset = None
            dur = None
        return CLAPNearest(
            song_path=song_path,
            similarity=float(sims[i]),
            song_id=song_id_from_path(song_path),
            offset_sec=offset,
            window_sec=dur,
        )


def _embed_song_windows(
    retriever: CLAPRetriever,
    song_path: str,
    *,
    embed_seconds: float,
    hop_seconds: float | None,
) -> tuple[list[np.ndarray], list[dict]]:
    """Return (embeddings, windows_meta) for one song.

    - ``hop_seconds is None`` (legacy): single embedding from song middle.
    - Otherwise: sliding windows of ``embed_seconds`` length every ``hop_seconds``.
    """
    if hop_seconds is None:
        emb = retriever.embed_file(song_path, max_seconds=embed_seconds, segment_from_middle=True)
        try:
            dur = max(0.0, _duration_sec(song_path))
        except Exception:
            dur = 0.0
        half = embed_seconds / 2.0
        start = max(0.0, dur / 2.0 - half) if dur > 0 else 0.0
        win = {
            "path": os.path.abspath(song_path),
            "start_sec": round(start, 3),
            "duration_sec": round(min(embed_seconds, dur) if dur > 0 else embed_seconds, 3),
        }
        return [emb], [win]

    dur = max(0.0, _duration_sec(song_path))
    if dur <= 0:
        return [], []
    starts: list[float] = []
    t = 0.0
    while t < dur:
        starts.append(t)
        t += hop_seconds
    if not starts:
        starts = [0.0]
    embs: list[np.ndarray] = []
    wins: list[dict] = []
    for s in starts:
        clip_len = min(embed_seconds, max(0.0, dur - s))
        if clip_len < min(2.0, embed_seconds * 0.5):
            continue
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        try:
            _ffmpeg_to_wav_48k_mono(song_path, wav_path, ss=s, duration_sec=clip_len)
            audio, sr = _read_wav_float_mono(wav_path)
        finally:
            if os.path.exists(wav_path):
                os.unlink(wav_path)
        if sr != 48000:
            raise ValueError(f"Expected 48000 Hz after ffmpeg, got {sr}")
        if audio.size == 0:
            continue
        vec = _embed_audio_array(retriever, audio)
        embs.append(vec)
        wins.append(
            {
                "path": os.path.abspath(song_path),
                "start_sec": round(float(s), 3),
                "duration_sec": round(float(clip_len), 3),
            }
        )
    return embs, wins


def _embed_audio_array(retriever: CLAPRetriever, audio: np.ndarray) -> np.ndarray:
    """Run the CLAP encoder on an already-decoded 48 kHz mono float array."""
    retriever._lazy_load()
    import torch

    with torch.no_grad():
        inputs = retriever._processor(
            audio=audio,
            sampling_rate=48000,
            return_tensors="pt",
        )
        inputs = {k: v.to(retriever._torch_device) for k, v in inputs.items()}
        if hasattr(retriever._model, "get_audio_features"):
            emb = retriever._model.get_audio_features(**inputs)
        else:
            out = retriever._model(**inputs)
            emb = getattr(out, "audio_embeds", None) or getattr(out, "pooler_output", None) or out[0]
        if hasattr(emb, "audio_embeds") and emb.audio_embeds is not None:
            emb = emb.audio_embeds
        elif hasattr(emb, "pooler_output") and emb.pooler_output is not None:
            emb = emb.pooler_output
        elif isinstance(emb, tuple) and len(emb) > 0:
            emb = emb[0]
        vec = emb.detach().cpu().numpy().astype(np.float32).reshape(-1)
    n = np.linalg.norm(vec)
    if n > 0:
        vec = vec / n
    return vec


def build_clap_index(
    songs_dir: str,
    output_json: str,
    *,
    model_name: str = DEFAULT_CLAP_MODEL,
    embed_seconds: float = DEFAULT_EMBED_SECONDS,
    device: str | None = None,
    extra_audio_paths: list[str] | None = None,
    hop_seconds: float | None = None,
) -> int:
    """Embed each audio file in ``songs_dir``. Saves JSON + .npy sidecar.

    - If ``hop_seconds is None`` (legacy): one embedding per song, from the middle.
    - If ``hop_seconds`` is set: sliding windows of ``embed_seconds`` every ``hop_seconds``,
      one row per window. The index can then report the best-matching window offset at
      query time (useful to locate the most similar part of a song that the fingerprint
      failed to recognize).

    ``extra_audio_paths``: additional files to embed (e.g. a track indexed for CLAP but omitted
    from the fingerprint ``songs_dir``). Deduped by ``os.path.realpath``.

    Returns the number of *songs* added to the index.
    """
    paths: list[str] = []
    seen_real: set[str] = set()

    def try_add(abs_path: str) -> None:
        abs_path = os.path.abspath(abs_path)
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"CLAP index audio not found: {abs_path}")
        key = os.path.realpath(abs_path)
        if key in seen_real:
            return
        seen_real.add(key)
        paths.append(abs_path)

    for name in sorted(os.listdir(songs_dir)):
        if not name.lower().endswith((".mp3", ".wav", ".flac", ".m4a")):
            continue
        try_add(os.path.join(songs_dir, name))
    for raw in sorted(extra_audio_paths or []):
        try_add(raw)

    retriever = CLAPRetriever(model_name=model_name, device=device)
    embeddings: list[np.ndarray] = []
    windows: list[dict] = []
    for p in paths:
        embs, wins = _embed_song_windows(
            retriever, p, embed_seconds=embed_seconds, hop_seconds=hop_seconds
        )
        embeddings.extend(embs)
        windows.extend(wins)

    mat = np.stack(embeddings, axis=0) if embeddings else np.zeros((0, 0), dtype=np.float32)
    npy_path = output_json.replace(".json", ".npy")
    np.save(npy_path, mat)

    meta = {
        "model_name": model_name,
        "embed_seconds": embed_seconds,
        "hop_seconds": hop_seconds,
        "segment_from_middle": hop_seconds is None,
        "paths": paths,  # deduped source songs
        "windows": windows,  # one entry per matrix row
        "embedding_npy": os.path.basename(npy_path),
    }
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return len(paths)


def load_clap_index(index_json: str) -> tuple[list[str], np.ndarray, dict]:
    """Return (row_paths, matrix, meta). ``row_paths[i]`` is the song for matrix row i.

    Legacy indices (no ``windows`` key) yield one row per song, so ``row_paths == meta["paths"]``.
    New indices include ``meta["windows"]`` with per-row ``start_sec`` / ``duration_sec``.
    """
    with open(index_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    base = os.path.dirname(index_json)
    npy_name = meta.get("embedding_npy") or Path(index_json).stem + ".npy"
    npy_path = os.path.join(base, npy_name)
    mat = np.load(npy_path)
    windows = meta.get("windows")
    if windows:
        row_paths = [w["path"] for w in windows]
    else:
        row_paths = meta["paths"]
    return row_paths, mat.astype(np.float32), meta
