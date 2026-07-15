#!/usr/bin/env python3
"""Smart mic pipeline con policy switch per segmento temporale della canzone.

Compatibile con `listen_smart_mic.py`, ma in `--song-to-policy` puoi usare:

- mapping classico: `thriller:4`
- mapping per range: `thriller@0-30:4,thriller@30-60:7,thriller@60-*:9`

Se l'offset della canzone e' disponibile, i range hanno precedenza.
In fallback resta valido il mapping classico `song -> policy`.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import listen_smart_mic as base  # noqa: E402


@dataclass(frozen=True)
class TimedPolicyRule:
    key: str
    start_s: float
    end_s: float | None
    policy_id: int


def _parse_bound(raw: str, *, allow_star: bool) -> float | None:
    token = raw.strip().lower()
    if allow_star and token in ("*", "inf", "infty", "infinite"):
        return None
    return float(token)


def parse_song_policy_rules(value: str) -> tuple[list[tuple[str, int]], list[TimedPolicyRule]]:
    plain: list[tuple[str, int]] = []
    timed: list[TimedPolicyRule] = []

    for raw in (value or "").split(","):
        item = raw.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid map entry '{item}'. Expected key:policy_id")

        left, pid_text = item.rsplit(":", 1)
        policy_id = int(pid_text.strip())
        left = left.strip().lower()

        if "@" not in left:
            plain.append((left, policy_id))
            continue

        key, range_part = left.split("@", 1)
        key = key.strip()
        range_part = range_part.strip()
        if not key:
            raise ValueError(f"Invalid timed map entry '{item}': empty key before '@'.")
        if "-" not in range_part:
            raise ValueError(
                f"Invalid timed map entry '{item}': expected range start-end after '@'."
            )
        start_text, end_text = range_part.split("-", 1)
        start_s = _parse_bound(start_text, allow_star=False)
        end_s = _parse_bound(end_text, allow_star=True)
        if start_s is None:
            raise ValueError(f"Invalid timed map entry '{item}': start cannot be '*'.")
        if start_s < 0:
            raise ValueError(f"Invalid timed map entry '{item}': start must be >= 0.")
        if end_s is not None and end_s <= start_s:
            raise ValueError(
                f"Invalid timed map entry '{item}': end ({end_s}) must be > start ({start_s})."
            )
        timed.append(TimedPolicyRule(key=key, start_s=start_s, end_s=end_s, policy_id=policy_id))

    return plain, timed


def _offset_in_range(offset_s: float, start_s: float, end_s: float | None) -> bool:
    if offset_s < start_s:
        return False
    if end_s is None:
        return True
    return offset_s < end_s


def resolve_policy_with_offset(
    *,
    song_path: str | None,
    song_id: str | None,
    offset_s: float | None,
    plain_mapping: list[tuple[str, int]],
    timed_rules: list[TimedPolicyRule],
) -> tuple[int | None, str]:
    candidates: list[str] = []
    if song_path:
        s = song_path.lower()
        candidates.append(s)
        candidates.append(Path(song_path).stem.lower())
    if song_id:
        candidates.append(song_id.lower())

    if offset_s is not None:
        for rule in timed_rules:
            for cand in candidates:
                if rule.key in cand and _offset_in_range(offset_s, rule.start_s, rule.end_s):
                    end_txt = "*" if rule.end_s is None else f"{rule.end_s:g}"
                    why = f"timed({rule.key}@{rule.start_s:g}-{end_txt})"
                    return rule.policy_id, why

    pid = base.resolve_policy(song_path, song_id, plain_mapping)
    if pid is not None:
        return pid, "plain(song)"
    return None, "no-match"


def _offline_playback_worker(audio_path: Path, start_delay_s: float = 0.0) -> None:
    if start_delay_s > 0:
        time.sleep(start_delay_s)

    ffplay = shutil.which("ffplay")
    if ffplay is not None:
        cmd = [
            ffplay,
            "-nodisp",
            "-autoexit",
            "-loglevel",
            "error",
            str(audio_path),
        ]
    else:
        print("[smart] offline playback skipped: ffplay not found in PATH.", flush=True)
        return

    try:
        subprocess.run(cmd, check=False)
    except Exception as exc:  # noqa: BLE001
        print(f"[smart] offline playback error: {exc}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    # Audio
    p.add_argument("--segment-seconds", type=float, default=5.0)
    p.add_argument("--sample-rate", type=int, default=48000)
    p.add_argument("--mic-channels", type=int, choices=(1, 2), default=1)
    p.add_argument("--sd-device", type=int, default=None)
    p.add_argument("--save-dir", default="runtime_chunks_live/smart_mic")
    p.add_argument(
        "--offline-mashup",
        default=None,
        help="Offline benchmark mode: read chunks from this audio file instead of the microphone.",
    )
    p.add_argument(
        "--offline-fast",
        action="store_true",
        help="When using --offline-mashup, do not pace to real-time (process chunks as fast as possible).",
    )
    p.add_argument(
        "--offline-playback",
        action="store_true",
        help="When using --offline-mashup, play the source file on the default speakers in a background thread.",
    )
    p.add_argument(
        "--offline-playback-delay",
        type=float,
        default=0.0,
        help="Optional delay before starting offline speaker playback.",
    )
    p.add_argument(
        "--command-delay-seconds",
        type=float,
        default=0.0,
        help=(
            "Delay before sending each policy command. Useful when offline speaker playback "
            "starts later than command processing."
        ),
    )
    p.add_argument(
        "--music-only",
        action="store_true",
        help="Skip AST/VAD and always run the music->Shazam->policy path (faster, avoids drift).",
    )
    p.add_argument("--speaker-device", type=int, default=None)

    # AST / VAD
    p.add_argument("--ast-model", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    p.add_argument("--ast-sample-rate", type=int, default=16000)
    p.add_argument("--ast-top-k", type=int, default=10)
    p.add_argument("--speech-margin", type=float, default=0.0,
                   help="Ramo speech se p(speech) >= p(music) + margin.")
    p.add_argument("--vad-threshold", type=float, default=0.3)
    p.add_argument("--vad-silence-fraction", type=float, default=0.03,
                   help="Se vad_frac < questa soglia il chunk e' considerato silenzio (skip).")
    p.add_argument("--music-min-score", type=float, default=0.20,
                   help="Se p(music) >= questa soglia, forziamo MUSIC ignorando la VAD.")

    # Shazam (music path)
    p.add_argument("--index-path", default="shazam/index.pkl")
    p.add_argument(
        "--song-to-policy",
        default=(
            # Timed rules (offset in seconds within the recognized song).
            # Policy IDs match `g1_switch_beyondmimic` / `g1_shazam_remote_listener`:
            #   2 bts2  3 dynamite  4 easy  5 swim  6 thriller  7 salsa
            #   8 gdance  9 salsa4  10 thriller_locked_waist
            "thriller@0-30:6,"       # intro/verse -> thriller
            "thriller@30-60:10,"    # middle     -> thriller_locked_waist
            "thriller@60-*:8,"      # outro      -> gdance
            "salsa@0-30:7,"          # early      -> salsa_tracking
            "salsa@30-*:9,"          # later      -> salsa4
            # Plain fallbacks (no time window): full-song policy.
            "dynamite:3,"
            "swim:5,"
            "bts:2,"
            "butter:2,"
            "gdance:8"
        ),
        help=(
            "Mapping song->policy. Supports per-section time ranges: "
            "thriller@0-30:6,thriller@30-60:10,thriller@60-*:8. "
            "Entries without '@' match the full song as fallback."
        ),
    )
    p.add_argument("--min-confidence", type=float, default=0.0)
    p.add_argument("--min-votes", type=int, default=10)
    p.add_argument("--send-repeat-after", type=float, default=8.0)
    p.add_argument(
        "--music-id",
        choices=("shazam", "clap", "shazam+clap", "rhythm"),
        default="shazam+clap",
        help=(
            "How to identify music chunks. "
            "'shazam' = fingerprint only; "
            "'clap' = CLAP only (stable for benchmark); "
            "'shazam+clap' = use Shazam when confident, else CLAP fallback."
            "'rhythm' = BPM/onset/tempogram features + lightweight classifier."
        ),
    )
    p.add_argument(
        "--clap-fallback",
        action="store_true",
        help="If Shazam match is low-confidence, fallback to CLAP similarity retrieval.",
    )
    p.add_argument(
        "--clap-index-json",
        default="shazam/clap_index.json",
        help="CLAP index JSON (has a .npy sidecar next to it).",
    )
    p.add_argument(
        "--clap-model",
        default=None,
        help="CLAP HF model name (default: index meta model_name / clap_fallback.DEFAULT_CLAP_MODEL).",
    )
    p.add_argument(
        "--clap-min-similarity",
        type=float,
        default=0.20,
        help="Minimum cosine similarity for CLAP fallback to be accepted.",
    )
    p.add_argument(
        "--clap-embed-seconds",
        type=float,
        default=10.0,
        help="Seconds of audio from the chunk to embed for CLAP.",
    )
    p.add_argument(
        "--clap-use-offset",
        action="store_true",
        help=(
            "Allow timed policy selection from CLAP window offsets. Default OFF because "
            "CLAP song retrieval is usable, but CLAP offsets are not reliable for localization."
        ),
    )
    p.add_argument(
        "--skip-clap-prewarm",
        action="store_true",
        help="Do not load CLAP HF weights until first embed (first chunk will be slow). Default: prewarm after init.",
    )
    p.add_argument(
        "--clap-class-map",
        default="",
        help=(
            "Optional CLAP class mode (category->policy). Format: "
            "'CLASS=stemA|stemB,OTHER=stemC|stemD'. "
            "When set, the listener will use CLAP similarity against class centroids "
            "instead of song-id mapping."
        ),
    )
    p.add_argument(
        "--clap-class-to-policy",
        default="",
        help="Mapping class->policy_id. Format: 'SALSA:7,BTS:2,OTHER:8'.",
    )
    p.add_argument(
        "--clap-class-min-margin",
        type=float,
        default=0.0,
        help="If >0, require (best_sim - second_best_sim) >= min_margin else no decision.",
    )
    p.add_argument("--rhythm-model", default="assets/models/rhythm_classifier.joblib", help="Rhythm classifier model path.")
    p.add_argument("--rhythm-min-proba", type=float, default=0.0, help="If >0, require max class proba >= threshold.")

    # OpenAI (speech path)
    p.add_argument("--openai-stt-model", default="gpt-4o-mini-transcribe")
    p.add_argument("--chat-model", default="gpt-4o-mini")
    p.add_argument("--chat-system-prompt", default="Reply briefly and naturally in English.")
    p.add_argument("--tts-model", default="gpt-4o-mini-tts")
    p.add_argument("--tts-voice", default="alloy")
    p.add_argument("--no-speak-response", action="store_true")
    p.add_argument("--speak-text-template", default="{ai_response}")
    p.add_argument("--min-transcript-chars", type=int, default=2)

    # Output format GESTI
    p.add_argument("--out-format", default="[GESTI,{tts_seconds:.2f}]",
                   help="Placeholder: {command},{clip_seconds},{api_seconds},{chat_api_seconds},{tts_seconds},{text}")
    p.add_argument("--command-name", default="GESTI")
    p.add_argument(
        "--max-gesti-seconds",
        type=float,
        default=15.0,
        help="Cap [GESTI,<seconds>] sent to robot (TTS may still play longer on speaker).",
    )

    # TCP
    p.add_argument("--tcp-mode", choices=("client", "server"), default="server")
    p.add_argument("--client-host", default="127.0.0.1")
    p.add_argument("--client-port", type=int, default=8765)
    p.add_argument("--server-host", default="0.0.0.0")
    p.add_argument("--server-port", type=int, default=8765)
    p.add_argument("--connect-timeout", type=float, default=2.0)
    p.add_argument(
        "--final-policy-id",
        type=int,
        default=1,
        help="When --offline-mashup reaches end-of-file, send this final [POLICY_SWITCH] id. Use a negative value to disable.",
    )
    p.add_argument(
        "--final-policy-settle-seconds",
        type=float,
        default=1.0,
        help="After sending the offline EOF final policy, wait this many seconds before closing TCP.",
    )
    p.add_argument(
        "--final-policy-repeat",
        type=int,
        default=3,
        help="How many times to send the offline EOF final policy command before closing.",
    )
    p.add_argument(
        "--final-policy-repeat-interval",
        type=float,
        default=0.2,
        help="Seconds between repeated offline EOF final policy sends.",
    )

    # Misc
    p.add_argument("--device", default=None, help="'cuda' o 'cpu'. Default: cuda se disponibile.")
    p.add_argument("--torch-dtype", default="float32", choices=("float32", "float16"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import torch
    except ImportError as e:
        raise SystemExit("Manca 'torch'. pip install torch") from e
    try:
        from transformers import pipeline
        from transformers.utils import logging as hf_logging
    except ImportError as e:
        raise SystemExit("Manca 'transformers'. pip install transformers") from e

    hf_logging.set_verbosity_error()
    warnings.filterwarnings("ignore", message=r".*[Ll]ogits [Pp]rocessor", category=UserWarning)
    warnings.filterwarnings("ignore", message=r".*A custom logits processor.*", category=UserWarning)

    index_path = Path(args.index_path)
    if not index_path.is_absolute():
        index_path = (ROOT / index_path).resolve()
    if not index_path.is_file():
        raise SystemExit(f"Shazam index not found: {index_path}")

    sys.path.insert(0, str((ROOT / "shazam").resolve()))
    from local_fingerprint import LocalShazamMatcher  # noqa: E402

    clf = None
    id2l: dict | None = None
    vad_model = None
    get_ts = None
    dev_name = "cpu"
    ast_sr = int(args.ast_sample_rate)
    if not bool(args.music_only):
        if args.device is None:
            device = 0 if torch.cuda.is_available() else -1
        elif args.device == "cpu":
            device = -1
        else:
            device = 0
        dev_name = "cuda" if device == 0 and torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if (dev_name == "cuda" and args.torch_dtype == "float16") else None

        print(f"[smart] loading AST {args.ast_model} on {dev_name} ...", flush=True)
        clf = pipeline("audio-classification", model=args.ast_model, device=device, torch_dtype=torch_dtype)
        mdl = getattr(clf, "model", None)
        cfg = getattr(mdl, "config", None) if mdl is not None else None
        id2l = dict(getattr(cfg, "id2label", None) or {}) if cfg is not None else None

        print("[smart] loading Silero VAD ...", flush=True)
        vad_model, get_ts = base.load_silero_vad()

    matcher = LocalShazamMatcher.load_index(str(index_path))
    plain_mapping, timed_mapping = parse_song_policy_rules(args.song_to_policy)

    clap_enabled = bool(getattr(args, "clap_fallback", False))
    clap_retriever = None
    clap_row_paths = None
    clap_matrix = None
    clap_windows = None
    if clap_enabled:
        try:
            from clap_fallback import CLAPRetriever, DEFAULT_CLAP_MODEL, load_clap_index  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"CLAP fallback requested but import failed: {exc}") from exc

        clap_index = Path(str(args.clap_index_json))
        if not clap_index.is_absolute():
            clap_index = (ROOT / clap_index).resolve()
        if not clap_index.is_file():
            raise SystemExit(f"CLAP index not found: {clap_index}")

        clap_row_paths, clap_matrix, meta = load_clap_index(str(clap_index))
        clap_windows = meta.get("windows") if isinstance(meta, dict) else None
        clap_model = str(args.clap_model or (meta.get("model_name") if isinstance(meta, dict) else None) or DEFAULT_CLAP_MODEL)

        clap_device = None
        if args.device in ("cuda", "cpu"):
            clap_device = str(args.device)
        clap_retriever = CLAPRetriever(model_name=clap_model, device=clap_device)
        if not bool(getattr(args, "skip_clap_prewarm", False)):
            print(f"[smart] prewarming CLAP {clap_model} ...", flush=True)
            clap_retriever.prewarm()
        print(
            f"[smart] CLAP fallback enabled model={clap_model} rows={len(clap_row_paths)} "
            f"min_sim={float(args.clap_min_similarity):.2f} use_offset={bool(args.clap_use_offset)}",
            flush=True,
        )
    if str(args.music_id) in ("clap", "shazam+clap") and not clap_enabled:
        raise SystemExit("--music-id requires --clap-fallback when using clap or shazam+clap.")
    if str(args.music_id) == "rhythm":
        # Ensure sklearn model is available.
        import joblib

        rhythm_model_path = Path(str(args.rhythm_model))
        if not rhythm_model_path.is_absolute():
            rhythm_model_path = (ROOT / rhythm_model_path).resolve()
        if not rhythm_model_path.is_file():
            raise SystemExit(f"--rhythm-model not found: {rhythm_model_path}")
        payload = joblib.load(str(rhythm_model_path))
        rhythm_model = payload.get("model")
        rhythm_features = payload.get("features") or []
        print(
            f"[smart] rhythm model loaded: {rhythm_model_path} features={len(rhythm_features)}",
            flush=True,
        )

    def _parse_clap_class_map(raw: str) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        for part in (raw or "").split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise SystemExit(f"Invalid --clap-class-map entry {part!r} (expected CLASS=stemA|stemB)")
            cname, stems_raw = part.split("=", 1)
            cname = cname.strip()
            stems = {Path(s.strip()).stem for s in stems_raw.split("|") if s.strip()}
            if not cname or not stems:
                raise SystemExit(f"Invalid --clap-class-map entry {part!r} (empty class or stems)")
            out[cname] = stems
        return out

    def _parse_clap_class_to_policy(raw: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for part in (raw or "").split(","):
            part = part.strip()
            if not part:
                continue
            if ":" not in part:
                raise SystemExit(f"Invalid --clap-class-to-policy entry {part!r} (expected CLASS:pid)")
            k, v = part.split(":", 1)
            out[k.strip()] = int(v.strip())
        return out

    clap_class_map = _parse_clap_class_map(str(getattr(args, "clap_class_map", "") or ""))
    clap_class_to_policy = _parse_clap_class_to_policy(str(getattr(args, "clap_class_to_policy", "") or ""))
    clap_class_centroids = {}
    clap_class_counts: dict[str, int] = {}
    if clap_class_map:
        if not clap_enabled or clap_row_paths is None or clap_matrix is None:
            raise SystemExit("--clap-class-map requires --clap-fallback (and a valid --clap-index-json).")
        if not clap_class_to_policy:
            raise SystemExit("--clap-class-map requires --clap-class-to-policy.")
        import numpy as np

        by_stem: dict[str, list] = {}
        for i, pth in enumerate(clap_row_paths):
            st = Path(pth).stem
            by_stem.setdefault(st, []).append(clap_matrix[i])
        for cname, stems in clap_class_map.items():
            vecs = []
            for st in stems:
                vecs.extend(by_stem.get(st, []))
            if not vecs:
                raise SystemExit(f"CLAP class {cname!r} has 0 rows in gallery (stems={sorted(stems)})")
            c = np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)
            n = float(np.linalg.norm(c))
            if n > 0:
                c = c / n
            clap_class_centroids[cname] = c
            clap_class_counts[cname] = len(vecs)
        print(
            "[smart] CLAP class mode enabled: "
            + ", ".join(
                [f"{k}(n={clap_class_counts[k]})=>pid={clap_class_to_policy.get(k)}" for k in clap_class_centroids]
            ),
            flush=True,
        )

    save_dir = Path(args.save_dir)
    if not save_dir.is_absolute():
        save_dir = (ROOT / save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    offline_path: Path | None = None
    offline_dur_s: float = 0.0
    if args.offline_mashup:
        offline_path = Path(str(args.offline_mashup))
        if not offline_path.is_absolute():
            offline_path = (ROOT / offline_path).resolve()
        if not offline_path.is_file():
            raise SystemExit(f"--offline-mashup not found: {offline_path}")

        r = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(offline_path),
            ],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            raise SystemExit(f"ffprobe failed for offline mashup:\n{r.stderr}")
        offline_dur_s = float(r.stdout.strip() or "0")
        print(f"[smart] OFFLINE mashup={offline_path} dur={offline_dur_s:.2f}s", flush=True)
        if bool(args.offline_playback) and not bool(args.offline_fast):
            threading.Thread(
                target=_offline_playback_worker,
                args=(offline_path, float(args.offline_playback_delay)),
                name="OfflineMashupPlayback",
                daemon=True,
            ).start()
            print(f"[smart] offline playback thread started: {offline_path.name}", flush=True)

    def _read_wav_f32_mono(path: Path) -> "base.np.ndarray":
        import wave
        import numpy as np

        with wave.open(str(path), "rb") as w:
            channels = int(w.getnchannels())
            sample_width = int(w.getsampwidth())
            sr = int(w.getframerate())
            nframes = int(w.getnframes())
            pcm = w.readframes(nframes)
        if sr <= 0:
            raise RuntimeError(f"Invalid wav sample rate: {sr}")
        if sample_width != 2:
            raise RuntimeError(f"Expected 16-bit wav, got sampwidth={sample_width}")
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)
        return audio

    if args.tcp_mode == "server":
        transport: base.GestureTcpClient | base.GestureTcpServer = base.GestureTcpServer(
            host=str(args.server_host), port=int(args.server_port), timeout_s=float(args.connect_timeout),
        )
        transport.start()
        print(f"[smart-server] listening on {args.server_host}:{args.server_port}", flush=True)
    else:
        transport = base.GestureTcpClient(
            host=str(args.client_host), port=int(args.client_port), timeout_s=float(args.connect_timeout),
        )
        print(f"[smart-client] target={args.client_host}:{args.client_port}", flush=True)

    print(f"[smart] Shazam index: {index_path}", flush=True)
    print(
        f"[smart] segment={args.segment_seconds}s mic_sr={args.sample_rate} ast_sr={args.ast_sample_rate} "
        f"speech_margin={args.speech_margin} vad_silence_fraction={args.vad_silence_fraction}",
        flush=True,
    )
    if timed_mapping:
        print(f"[smart] timed policy rules loaded: {len(timed_mapping)}", flush=True)

    last_policy_sent: int | None = None
    last_send_ts: float = 0.0
    offline_eof_reached = False

    try:
        idx = 0
        while True:
            t0 = time.monotonic()
            chunk_started_wall = base.wall_clock_ts()
            if offline_path is not None:
                ss = float(idx) * float(args.segment_seconds)
                if ss >= max(0.0, float(offline_dur_s)):
                    offline_eof_reached = True
                    break
                if idx == 0:
                    t0_end = min(float(args.segment_seconds), max(0.0, float(offline_dur_s)))
                    print(
                        f"{chunk_started_wall} [INFO] [robojudo.listener.timed_policy] "
                        f"offline_first_audio_window t_file=[0.000,{t0_end:.3f})s wall={chunk_started_wall} "
                        f"(start processing segment at file time 0; no speaker play)",
                        flush=True,
                    )
                clip_path = save_dir / f"chunk_{idx:06d}.wav"
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    f"{ss:.6f}",
                    "-t",
                    f"{float(args.segment_seconds):.6f}",
                    "-i",
                    str(offline_path),
                    "-ac",
                    "1",
                    "-ar",
                    str(int(args.sample_rate)),
                    "-vn",
                    "-acodec",
                    "pcm_s16le",
                    str(clip_path),
                ]
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode != 0:
                    print(f"[smart] offline ffmpeg error: {r.stderr}", flush=True)
                    idx += 1
                    continue
                raw = _read_wav_f32_mono(clip_path)
            else:
                base.wait_tts_idle(verbose=bool(args.verbose))
                raw = base.record_mono_f32(
                    sample_rate=int(args.sample_rate),
                    channels=int(args.mic_channels),
                    duration_s=float(args.segment_seconds),
                    sd_device=args.sd_device,
                )
                if args.verbose:
                    peak, rms = base.mono_peak_rms(raw)
                    print(f"[smart] mic level peak={peak:.4f} rms={rms:.4f}", flush=True)
                clip_path = save_dir / f"chunk_{idx:06d}.wav"
                try:
                    base.write_wav_s16le_mono(clip_path, raw, int(args.sample_rate))
                except OSError as e:
                    print(f"[smart] wav write err: {e!r}", flush=True)

            audio_ast = base.resample_f32(raw, int(args.sample_rate), ast_sr)
            if bool(args.music_only):
                p_sp, p_mu, vad_frac = 0.0, 1.0, 1.0
                decision = "MUSIC(music-only)"
                branch = "music"
            else:
                assert clf is not None and vad_model is not None and get_ts is not None
                ast_in = {"raw": audio_ast, "sampling_rate": ast_sr}
                try:
                    out_list = clf(ast_in, top_k=int(args.ast_top_k))
                except TypeError:
                    out_list = clf(ast_in)
                if not isinstance(out_list, list):
                    out_list = [out_list]  # type: ignore[assignment]
                out_list = base._expand_id2label(out_list, id2l)
                p_sp, p_mu = base._label_scores_ast(out_list, args.verbose)

                vad_frac = base.silero_speech_fraction(
                    audio_ast, vad_model, get_ts,
                    threshold=float(args.vad_threshold),
                    sample_rate=ast_sr,
                )

                if p_mu >= float(args.music_min_score) and p_mu > p_sp:
                    decision = "MUSIC"
                    branch = "music"
                elif vad_frac < float(args.vad_silence_fraction):
                    decision = "SKIP(silence)"
                    branch = "skip"
                elif p_sp >= p_mu + float(args.speech_margin):
                    decision = "SPEECH"
                    branch = "speech"
                else:
                    decision = "MUSIC"
                    branch = "music"

            if args.verbose:
                print(
                    f"[smart] chunk={idx} p_speech={p_sp:.2f} p_music={p_mu:.2f} "
                    f"vad_frac={vad_frac:.2f} -> {decision}",
                    flush=True,
                )

            if branch == "music":
                music_id_mode = str(args.music_id)
                policy_id = None
                policy_reason = "no-match"
                song_name = "unknown"
                conf_txt = "0.00"
                votes_txt = "0"
                offset_txt = "None"

                # 1) Shazam path (if enabled)
                shazam_ok = False
                if music_id_mode in ("shazam", "shazam+clap"):
                    result = matcher.match(str(clip_path))
                    policy_id, policy_reason = resolve_policy_with_offset(
                        song_path=result.song_path,
                        song_id=result.song_id,
                        offset_s=result.offset_sec,
                        plain_mapping=plain_mapping,
                        timed_rules=timed_mapping,
                    )
                    song_name = Path(result.song_path).stem if result.song_path else "unknown"
                    conf_txt = f"{float(result.confidence):.2f}"
                    votes_txt = str(int(result.votes))
                    offset_txt = str(result.offset_sec)
                    shazam_ok = not (
                        policy_id is None
                        or float(result.confidence) < float(args.min_confidence)
                        or int(result.votes) < int(args.min_votes)
                    )

                # 2) CLAP path (if enabled)
                if (music_id_mode == "clap" or (music_id_mode == "shazam+clap" and not shazam_ok)):
                    nearest = None
                    try:
                        assert clap_retriever is not None and clap_row_paths is not None and clap_matrix is not None
                        nearest = clap_retriever.nearest(
                            str(clip_path),
                            clap_row_paths,
                            clap_matrix,
                            max_seconds=float(args.clap_embed_seconds),
                            windows=clap_windows,
                        )
                    except Exception as exc:  # noqa: BLE001
                        if args.verbose:
                            print(f"[smart] CLAP error: {exc}", flush=True)

                    if nearest is not None and float(nearest.similarity) >= float(args.clap_min_similarity):
                        # If class mode is enabled, choose a category by centroid similarity and map to policy_id.
                        if clap_class_centroids:
                            import numpy as np

                            q = clap_retriever.embed_file(
                                str(clip_path),
                                max_seconds=float(args.clap_embed_seconds),
                                segment_from_middle=False,
                            )
                            scored = sorted(
                                ((cname, float(np.dot(c, q))) for cname, c in clap_class_centroids.items()),
                                key=lambda kv: kv[1],
                                reverse=True,
                            )
                            cname, csim = scored[0]
                            second = scored[1][1] if len(scored) > 1 else -1.0
                            margin = float(csim) - float(second)
                            if margin < float(args.clap_class_min_margin):
                                policy_id = None
                                song_name = "unknown"
                                policy_reason = f"clap-class(ambig margin<{float(args.clap_class_min_margin):.3f})"
                                conf_txt = f"{float(csim):.3f}"
                                votes_txt = "-"
                                offset_txt = "None"
                            else:
                                policy_id = clap_class_to_policy.get(cname)
                                song_name = cname
                                policy_reason = f"clap-class({cname} sim={csim:.3f} margin={margin:.3f})"
                                conf_txt = f"{float(csim):.3f}"
                                votes_txt = "-"
                                offset_txt = "None"
                        else:
                            policy_id, policy_reason = resolve_policy_with_offset(
                                song_path=nearest.song_path,
                                song_id=nearest.song_id,
                                offset_s=nearest.offset_sec if bool(args.clap_use_offset) else None,
                                plain_mapping=plain_mapping,
                                timed_rules=timed_mapping,
                            )
                            song_name = Path(nearest.song_path).stem if nearest.song_path else "unknown"
                            policy_reason = f"{policy_reason}|clap(sim={nearest.similarity:.3f})"
                            conf_txt = f"{float(nearest.similarity):.3f}"
                            votes_txt = "-"
                            offset_txt = str(nearest.offset_sec) if bool(args.clap_use_offset) else "None"

                # 3) Rhythm classifier path
                if music_id_mode == "rhythm":
                    from scripts.rhythm_features import extract_rhythm_features
                    import numpy as np

                    feats = extract_rhythm_features(raw, sr=int(args.sample_rate))
                    vec = feats.as_vector().reshape(1, -1)
                    proba = None
                    pred_class = None
                    try:
                        if hasattr(rhythm_model, "predict_proba"):
                            proba = rhythm_model.predict_proba(vec)[0]
                            classes = list(getattr(rhythm_model, "classes_", []))
                            j = int(np.argmax(proba)) if len(proba) else 0
                            pred_class = str(classes[j]) if classes else None
                            pmax = float(proba[j]) if len(proba) else 0.0
                        else:
                            pred_class = str(rhythm_model.predict(vec)[0])
                            pmax = 1.0
                    except Exception as exc:  # noqa: BLE001
                        if args.verbose:
                            print(f"[smart] rhythm error: {exc}", flush=True)
                        pred_class = None
                        pmax = 0.0

                    if pred_class is not None and float(pmax) >= float(args.rhythm_min_proba):
                        # Map predicted class name through clap_class_to_policy if provided, else song_to_policy.
                        if clap_class_to_policy:
                            policy_id = clap_class_to_policy.get(pred_class)
                        song_name = pred_class
                        policy_reason = f"rhythm(class={pred_class} p={pmax:.3f})"
                        conf_txt = f"{pmax:.3f}"
                        votes_txt = "-"
                        offset_txt = "None"

                if args.verbose:
                    _nw = base.wall_clock_ts()
                    _el = time.monotonic() - t0
                    print(
                        f"{_nw} [INFO] [robojudo.listener.timed_policy] "
                        f"Listener music predict chunk={idx} policy_id={policy_id} song={song_name!r} "
                        f"reason={policy_reason} conf={conf_txt} votes={votes_txt} offset={offset_txt} "
                        f"started_wall={chunk_started_wall} now_wall={_nw} elapsed_s={_el:.3f}",
                        flush=True,
                    )
                if policy_id is None:
                    if args.verbose:
                        print(
                            f"[smart] music skip song={song_name} conf={conf_txt} "
                            f"votes={votes_txt} offset={offset_txt} policy={policy_id}",
                            flush=True,
                        )
                else:
                    now = time.monotonic()
                    if (last_policy_sent is not None and policy_id == last_policy_sent
                            and (now - last_send_ts) < float(args.send_repeat_after)):
                        if args.verbose:
                            _nw = base.wall_clock_ts()
                            print(
                                f"{_nw} [INFO] [robojudo.listener.timed_policy] "
                                f"Listener music throttled policy_id={policy_id} reason={policy_reason} "
                                f"started_wall={chunk_started_wall} now_wall={_nw} "
                                f"elapsed_s={time.monotonic() - t0:.3f}",
                                flush=True,
                            )
                            print(f"[smart] throttled policy={policy_id} reason={policy_reason}", flush=True)
                    else:
                        line = f"[POLICY_SWITCH],{policy_id}"
                        if float(args.command_delay_seconds) > 0.0:
                            time.sleep(float(args.command_delay_seconds))
                        if args.dry_run:
                            _nw = base.wall_clock_ts()
                            print(
                                f"{_nw} [INFO] [robojudo.listener.timed_policy] "
                                f"Listener music dry_run line={line!r} started_wall={chunk_started_wall} "
                                f"now_wall={_nw} elapsed_s={time.monotonic() - t0:.3f}",
                                flush=True,
                            )
                            print(
                                f"[smart] chunk={idx} DRY-RUN {line} song={song_name} conf={conf_txt} "
                                f"votes={votes_txt} offset={offset_txt} reason={policy_reason}",
                                flush=True,
                            )
                            last_policy_sent = policy_id
                            last_send_ts = now
                        elif base._send_tcp_line(transport, line):
                            _nw = base.wall_clock_ts()
                            _el = time.monotonic() - t0
                            # Timestamps like rl_multi_policy_pipeline._log_policy_segment_wallclock
                            print(
                                f"{_nw} [INFO] [robojudo.listener.timed_policy] "
                                f"Policy segment elapsed id={policy_id} MusicListener@{song_name} "
                                f"started_wall={chunk_started_wall} now_wall={_nw} elapsed_s={_el:.3f}",
                                flush=True,
                            )
                            print(
                                f"[smart] chunk={idx} sent {line} song={song_name} conf={conf_txt} "
                                f"votes={votes_txt} offset={offset_txt} reason={policy_reason}",
                                flush=True,
                            )
                            last_policy_sent = policy_id
                            last_send_ts = now

            elif branch == "speech":
                try:
                    text, stt_s = base.transcribe_with_openai(clip_path, str(args.openai_stt_model))
                except Exception as exc:  # noqa: BLE001
                    print(f"[smart] openai stt error: {exc}", flush=True)
                    idx += 1
                    continue
                preview = text[:80] + ("..." if len(text) > 80 else "")
                if args.verbose:
                    print(f"[smart] openai stt_s={stt_s:.2f} text={preview!r}", flush=True)
                if len((text or "").strip()) < int(args.min_transcript_chars):
                    if args.verbose:
                        print("[smart] skip: transcript too short", flush=True)
                    idx += 1
                    continue

                override_line, forced_reply = base._special_routing_from_text(text)
                chat_api_seconds = 0.0
                ai_response_text = forced_reply or ""
                if forced_reply is None:
                    try:
                        ai_response_text, chat_api_seconds = base.generate_ai_response_text(
                            user_text=text or "silenzio",
                            model=str(args.chat_model),
                            system_prompt=str(args.chat_system_prompt),
                        )
                    except Exception as exc:  # noqa: BLE001
                        ai_response_text = "Scusa, c'e' stato un errore."
                        print(f"[smart] ai-reply error: {exc}", flush=True)

                tts_seconds = 0.0
                tts_audio = None
                tts_sr = 0
                if not bool(args.no_speak_response):
                    spoken_text = str(args.speak_text_template).format(
                        text=text or "silenzio", ai_response=ai_response_text,
                    )
                    try:
                        tts_wav, _ = base.synthesize_tts_wav_with_openai(
                            spoken_text, model=str(args.tts_model), voice=str(args.tts_voice),
                        )
                        tts_audio, tts_sr = base.decode_wav_bytes(tts_wav)
                        tts_seconds = base.wav_duration_seconds_from_decoded(tts_audio, tts_sr)
                        if tts_seconds <= 0.0 or tts_seconds > 120.0:
                            words = len((spoken_text or "").split())
                            tts_seconds = max(0.8, words / 2.7)
                    except Exception as exc:  # noqa: BLE001
                        tts_audio = None
                        print(f"[smart] tts error: {exc}", flush=True)

                line = base.format_line(
                    args.out_format,
                    command=str(args.command_name),
                    clip_seconds=float(args.segment_seconds),
                    api_seconds=stt_s,
                    chat_api_seconds=chat_api_seconds,
                    tts_seconds=min(tts_seconds, float(args.max_gesti_seconds)),
                    text=text,
                )
                if override_line is not None:
                    line = override_line

                if args.dry_run:
                    print(f"[smart] DRY-RUN line={line!r} text={preview!r}", flush=True)
                else:
                    if base._send_tcp_line(transport, line):
                        print(f"[smart] sent line={line!r} text={preview!r}", flush=True)

                if tts_audio is not None and not bool(args.no_speak_response):
                    base.play_audio_array_background(tts_audio, tts_sr, speaker_device=args.speaker_device)

            if args.verbose:
                _end_w = base.wall_clock_ts()
                _end_el = time.monotonic() - t0
                print(
                    f"{_end_w} [INFO] [robojudo.listener.timed_policy] "
                    f"Listener chunk wall chunk={idx} started_wall={chunk_started_wall} now_wall={_end_w} "
                    f"elapsed_s={_end_el:.3f}",
                    flush=True,
                )
                print(f"[smart] chunk wall time {_end_el:.2f}s", flush=True)
            # In offline mode, pace to real-time so pipeline timings match audio playback.
            if offline_path is not None and not bool(args.offline_fast):
                spent = time.monotonic() - t0
                remaining = float(args.segment_seconds) - float(spent)
                if remaining > 0:
                    time.sleep(min(remaining, float(args.segment_seconds)))
            idx += 1
        if offline_path is not None and offline_eof_reached and int(args.final_policy_id) >= 0:
            line = f"[POLICY_SWITCH],{int(args.final_policy_id)}"
            end_wall = base.wall_clock_ts()
            if args.dry_run:
                print(
                    f"{end_wall} [INFO] [robojudo.listener.timed_policy] offline EOF -> DRY-RUN final line={line!r}",
                    flush=True,
                )
            else:
                repeats = max(1, int(args.final_policy_repeat))
                interval_s = max(0.0, float(args.final_policy_repeat_interval))
                if float(args.command_delay_seconds) > 0.0:
                    time.sleep(float(args.command_delay_seconds))
                sent_count = 0
                for i in range(repeats):
                    if base._send_tcp_line(transport, line):
                        sent_count += 1
                    if i + 1 < repeats and interval_s > 0:
                        time.sleep(interval_s)
                print(
                    f"{end_wall} [INFO] [robojudo.listener.timed_policy] "
                    f"offline EOF -> sent final line={line!r} repeats={sent_count}/{repeats}",
                    flush=True,
                )
                settle_s = max(0.0, float(args.final_policy_settle_seconds))
                if settle_s > 0:
                    time.sleep(settle_s)
    except KeyboardInterrupt:
        print("\n[smart] stopped", flush=True)
    finally:
        transport.close()


if __name__ == "__main__":
    main()
