#!/usr/bin/env python3
"""Smart mic pipeline: AST (speech vs music) -> OpenAI(GESTI) oppure Shazam(POLICY_SWITCH).

Flusso per ogni chunk microfono (default 5s):

  1. Registra WAV dal mic.
  2. AST (MIT AudioSet 527 classi) + Silero VAD sanity-check.
  3. Se il contenuto e' **speech**:
       - OpenAI Whisper STT (cloud)
       - regole speciali ("violin", "67", "what can you do?")
       - chat reply OpenAI + TTS su speaker
       - invio TCP della riga `[GESTI,<tts_seconds>]` (o override speciale come `[Violin]` / `[67]`).
  4. Se il contenuto e' **music**:
       - Shazam locale sull'index
       - invio TCP `[POLICY_SWITCH],<policy_id>` con mapping canzone->policy.
  5. Se e' silenzio: skip.

Richiede le dipendenze di entrambi gli script originali:
  ``torch``, ``transformers``, ``scipy``, ``sounddevice``, ``numpy``, ``openai``.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import warnings
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]


def _ensure_torch_hub_writable() -> None:
    """Use a writable TORCH_HOME (default ~/.cache/torch may be root-owned on some setups)."""
    candidates: list[Path] = []
    env_home = os.environ.get("TORCH_HOME", "").strip()
    if env_home:
        candidates.append(Path(env_home))
    candidates.append(Path.home() / ".cache" / "torch")
    candidates.append(ROOT / ".cache" / "torch")

    for base in candidates:
        hub = base / "hub"
        try:
            hub.mkdir(parents=True, exist_ok=True)
            probe = hub / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            os.environ["TORCH_HOME"] = str(base)
            return
        except OSError:
            continue


def wall_clock_ts() -> str:
    """Same wall-clock string as `robojudo.pipeline.rl_multi_policy_pipeline._wall_clock_ts` (for log alignment)."""
    now = datetime.now()
    return now.strftime("%m-%d %H:%M:%S.") + f"{now.microsecond // 1000:03d}"


# ---------------------------------------------------------------------------
# Util: OpenAI key
# ---------------------------------------------------------------------------
def resolve_openai_api_key() -> str:
    env_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if env_key:
        return env_key
    secrets_path = ROOT / "secrets.json"
    if secrets_path.is_file():
        try:
            payload = json.loads(secrets_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                file_key = str(payload.get("openai_key", "")).strip()
                if file_key:
                    return file_key
        except (OSError, json.JSONDecodeError):
            pass
    raise SystemExit("OpenAI key not found. Set OPENAI_API_KEY or add secrets.json with openai_key.")


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def record_mono_f32(sample_rate: int, channels: int, duration_s: float, sd_device: int | None) -> np.ndarray:
    frames = int(sample_rate * duration_s)
    last_mono = np.zeros(frames, dtype=np.float32)
    for attempt in range(3):
        audio = sd.rec(frames, samplerate=sample_rate, channels=channels, dtype="float32", device=sd_device)
        sd.wait()
        if channels == 1:
            mono = np.asarray(audio[:, 0] if audio.ndim == 2 else audio, dtype=np.float32)
        else:
            mono = np.mean(audio, axis=1, dtype=np.float32)
        mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)
        mono = np.clip(mono, -1.0, 1.0)
        last_mono = mono
        peak = float(np.max(np.abs(mono))) if mono.size else 0.0
        if peak > 1e-5:
            return mono
        if attempt < 2:
            print(
                f"[smart] mic silent/invalid peak={peak:.2e} attempt={attempt + 1}/3, resetting capture",
                flush=True,
            )
            try:
                sd.stop()
            except Exception:
                pass
            time.sleep(0.25)
    return last_mono


def mono_peak_rms(mono: np.ndarray) -> tuple[float, float]:
    if mono.size == 0:
        return 0.0, 0.0
    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(np.square(mono.astype(np.float64)))))
    return peak, rms


def _speaker_device_name(speaker_device: int | None) -> str | None:
    if speaker_device is None:
        return None
    try:
        name = str(sd.query_devices(speaker_device).get("name", "")).strip()
        return name or None
    except Exception:
        return None


def _play_wav_file_external(path: Path, speaker_device: int | None) -> None:
    """Play WAV without sounddevice (avoids sd.play/sd.rec conflict on PipeWire)."""
    dev_name = _speaker_device_name(speaker_device)
    candidates: list[list[str]] = []
    if dev_name:
        candidates.extend(
            [
                ["paplay", f"--device={dev_name}", str(path)],
                ["pw-play", "--target", dev_name, str(path)],
            ]
        )
    candidates.append(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)])

    last_err = ""
    for cmd in candidates:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180.0)
        except FileNotFoundError:
            continue
        except subprocess.TimeoutExpired:
            last_err = f"timeout: {' '.join(cmd)}"
            continue
        if proc.returncode == 0:
            return
        err = (proc.stderr or proc.stdout or "").strip()
        last_err = f"{' '.join(cmd)} rc={proc.returncode} {err[:200]}"
    raise RuntimeError(last_err or "no external audio player found (paplay/pw-play/ffplay)")


def play_audio_array(audio: np.ndarray, sample_rate: int, speaker_device: int | None) -> float:
    t0 = time.perf_counter()
    mono = np.asarray(audio, dtype=np.float32)
    if mono.ndim == 2:
        mono = mono.mean(axis=1)
    play_sr = int(sample_rate)
    if speaker_device is not None:
        try:
            dev_sr = int(float(sd.query_devices(speaker_device)["default_samplerate"]))
            if dev_sr > 0 and dev_sr != play_sr:
                mono = resample_f32(mono, play_sr, dev_sr)
                play_sr = dev_sr
        except Exception:
            pass
    tmp_path = Path(tempfile.mkstemp(suffix=".wav", prefix="smart_tts_")[1])
    try:
        write_wav_s16le_mono(tmp_path, mono, play_sr)
        _play_wav_file_external(tmp_path, speaker_device)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
    return max(0.0, time.perf_counter() - t0)


def write_wav_s16le_mono(path: Path, mono_f32: np.ndarray, sample_rate: int) -> None:
    pcm = (np.clip(mono_f32, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm.tobytes())


def resample_f32(mono: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return mono.astype(np.float32, copy=False)
    n_out = int(round(mono.shape[0] * target_sr / float(orig_sr)))
    return signal.resample(mono, n_out).astype(np.float32)


def decode_wav_bytes(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        channels = w.getnchannels()
        sample_width = w.getsampwidth()
        sample_rate = w.getframerate()
        nframes = w.getnframes()
        pcm = w.readframes(nframes)
    if sample_width == 2:
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 1:
        audio = (np.frombuffer(pcm, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 4:
        audio = np.frombuffer(pcm, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")
    if channels > 1:
        audio = audio.reshape(-1, channels)
    return audio, sample_rate


def wav_duration_seconds_from_decoded(audio: np.ndarray, sample_rate: int) -> float:
    if sample_rate <= 0:
        return 0.0
    nframes = int(audio.shape[0]) if audio.ndim >= 1 else 0
    if nframes <= 0:
        return 0.0
    return float(nframes) / float(sample_rate)


_tts_thread: threading.Thread | None = None


def play_audio_array_background(
    audio: np.ndarray, sample_rate: int, speaker_device: int | None
) -> None:
    """Play TTS in a background thread (paplay/ffplay, not sd.play)."""
    global _tts_thread

    def _run() -> None:
        try:
            playback_s = play_audio_array(audio, sample_rate, speaker_device)
            print(f"[smart] speaker playback ok tts_s={playback_s:.2f}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[smart] speaker err: {exc}", flush=True)

    if _tts_thread is not None and _tts_thread.is_alive():
        _tts_thread.join(timeout=0.5)
    _tts_thread = threading.Thread(target=_run, name="smart-tts-playback", daemon=True)
    _tts_thread.start()


def is_tts_playing() -> bool:
    return _tts_thread is not None and _tts_thread.is_alive()


def wait_tts_idle(poll_s: float = 0.05, verbose: bool = False) -> None:
    if not is_tts_playing():
        return
    if verbose:
        print("[smart] waiting for TTS playback to finish before next mic chunk...", flush=True)
    while is_tts_playing():
        time.sleep(poll_s)
    if verbose:
        print("[smart] TTS finished, resuming mic capture", flush=True)


# ---------------------------------------------------------------------------
# AST (AudioSet) label aggregation
# ---------------------------------------------------------------------------
_SPEECH_KEYS: tuple[str, ...] = (
    "speech",
    "conversation",
    "narration",
    "monologue",
    "whispering",
    "chatter",
    "babbling",
)
_MUSIC_KEYS: tuple[str, ...] = (
    "music", "singing", "song", "vocal music", "choir", "chant", "rapping",
    "yodeling", "humming", "musical instrument", "guitar", "bass guitar",
    "drum", "piano", "keyboard", "synthesizer", "saxophone", "violin",
    "trumpet", "orchestra", "brass instrument", "string instrument",
    "percussion", "bell", "a capella",
)
_SPEECH_EXCLUDE: tuple[str, ...] = ("speech synthesizer",)


def _keyword_hit(label: str, keys: tuple[str, ...]) -> bool:
    return any(k in label for k in keys)


def _expand_id2label(outputs: list[dict], id2label: dict | None) -> list[dict]:
    if not id2label:
        return outputs
    out: list[dict] = []
    for o in outputs:
        lab = str(o.get("label", ""))
        low = lab.lower()
        if low.startswith("label_") and " " not in low:
            try:
                idx = int(low.split("_", 1)[-1])
                raw = id2label.get(idx)
                if raw is None and str(idx) in id2label:
                    raw = id2label[str(idx)]
                mapped = str(raw if raw is not None else lab)
            except ValueError:
                mapped = lab
        else:
            mapped = lab
        out.append({"label": mapped.lower(), "score": o["score"]})
    return out


def _label_scores_ast(outputs: list[dict], verbose: bool) -> tuple[float, float]:
    by: dict[str, float] = {}
    for o in outputs:
        lab = str(o.get("label", "")).lower().strip()
        if not lab:
            continue
        by[lab] = max(by.get(lab, 0.0), float(o.get("score", 0.0)))
    p_sp = 0.0
    p_mu = 0.0
    for lab, sc in by.items():
        if _keyword_hit(lab, _SPEECH_EXCLUDE):
            p_mu = max(p_mu, sc)
            continue
        if _keyword_hit(lab, _SPEECH_KEYS):
            p_sp = max(p_sp, sc)
        elif _keyword_hit(lab, _MUSIC_KEYS):
            p_mu = max(p_mu, sc)
    if verbose and by:
        tops = sorted(by.items(), key=lambda x: x[1], reverse=True)[:6]
        print("[smart] ast top:", ", ".join(f"{l}={s:.3f}" for l, s in tops), flush=True)
    return p_sp, p_mu


# ---------------------------------------------------------------------------
# Silero VAD (sanity-check silenzio)
# ---------------------------------------------------------------------------
def load_silero_vad():
    import torch
    _ensure_torch_hub_writable()
    try:
        import torchaudio  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "Silero VAD richiede 'torchaudio'. Install: pip install torchaudio"
        ) from e
    try:
        model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
    except TypeError:
        model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad")
    get_speech_timestamps = utils[0]
    model.eval()
    return model, get_speech_timestamps


def silero_speech_fraction(mono_16k: np.ndarray, model, get_ts, *, threshold: float, sample_rate: int) -> float:
    import torch
    n = int(mono_16k.shape[0])
    if n < 1:
        return 0.0
    w = torch.from_numpy(np.ascontiguousarray(mono_16k, dtype=np.float32))
    ts = get_ts(w, model, sampling_rate=sample_rate, threshold=threshold,
                min_speech_duration_ms=80, min_silence_duration_ms=80)
    covered = 0.0
    for t in ts:
        covered += float(t["end"] - t["start"])
    return min(1.0, covered / float(n))


# ---------------------------------------------------------------------------
# Shazam mapping
# ---------------------------------------------------------------------------
def parse_song_map(value: str) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for raw in (value or "").split(","):
        item = raw.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid map entry '{item}'. Expected key:policy_id")
        key, pid = item.split(":", 1)
        out.append((key.strip().lower(), int(pid)))
    return out


def resolve_policy(song_path: str | None, song_id: str | None, mapping: list[tuple[str, int]]) -> int | None:
    candidates: list[str] = []
    if song_path:
        candidates.append(song_path.lower())
        candidates.append(Path(song_path).stem.lower())
    if song_id:
        candidates.append(song_id.lower())
    for key, policy_id in mapping:
        for cand in candidates:
            if key in cand:
                return policy_id
    return None


# ---------------------------------------------------------------------------
# OpenAI helpers
# ---------------------------------------------------------------------------
def transcribe_with_openai(wav_path: Path, model: str) -> tuple[str, float]:
    from openai import OpenAI
    client = OpenAI(api_key=resolve_openai_api_key())
    t0 = time.perf_counter()
    with wav_path.open("rb") as fh:
        result = client.audio.transcriptions.create(model=model, file=fh)
    elapsed = time.perf_counter() - t0
    text = (getattr(result, "text", "") or "").strip()
    return text, elapsed


def generate_ai_response_text(user_text: str, model: str, system_prompt: str) -> tuple[str, float]:
    from openai import OpenAI
    client = OpenAI(api_key=resolve_openai_api_key())
    t0 = time.perf_counter()
    result = client.responses.create(model=model, instructions=system_prompt, input=user_text)
    elapsed = time.perf_counter() - t0
    out = (getattr(result, "output_text", "") or "").strip()
    if not out:
        out = "Non ho capito bene, puoi ripetere?"
    return out, elapsed


def synthesize_tts_wav_with_openai(text: str, model: str, voice: str) -> tuple[bytes, float]:
    from openai import OpenAI
    client = OpenAI(api_key=resolve_openai_api_key())
    t0 = time.perf_counter()
    try:
        response = client.audio.speech.create(model=model, voice=voice, input=text, response_format="wav")
    except TypeError:
        response = client.audio.speech.create(model=model, voice=voice, input=text)
    wav_bytes = response.read()
    elapsed = time.perf_counter() - t0
    return wav_bytes, elapsed


# ---------------------------------------------------------------------------
# Regole speciali (GESTI override)
# ---------------------------------------------------------------------------
def _normalize_text(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _special_routing_from_text(text: str) -> tuple[str | None, str | None]:
    norm = _normalize_text(text)
    if not norm:
        return None, None
    if re.search(r"\bwhat\b.*\bcan\b.*\bdo\b", norm) or re.search(r"\bwhat\b.*\byou\b.*\bdo\b", norm):
        return None, "i can do Violin, and 67 want to try?"
    has_violin = bool(re.search(r"\bviolin(?:o|i)?\b", norm) or re.search(r"\bviolon", norm))
    has_67 = bool(
        re.search(r"\b67\b", norm)
        or re.search(r"\b6\s*[/\-]\s*7\b", norm)
        or re.search(r"\bsix[ -]?seven\b", norm)
        or re.search(r"\bsixty[ -]?seven\b", norm)
        or re.search(r"\bsei\s+sette\b", norm)
        or re.search(r"\bsessanta[ -]?sette\b", norm)
    )
    if has_violin:
        return "[Violin]", "Okay, I can do Violin."
    if has_67:
        return "[67]", "Okay, I can do 67."
    return None, None


# ---------------------------------------------------------------------------
# TCP transport
# ---------------------------------------------------------------------------
class GestureTcpClient:
    def __init__(self, host: str, port: int, timeout_s: float):
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.sock: socket.socket | None = None

    def _connect(self) -> None:
        self.close()
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
        sock.settimeout(self.timeout_s)
        self.sock = sock

    def send_line(self, line: str) -> None:
        payload = (line.rstrip("\n") + "\n").encode("utf-8")
        if self.sock is None:
            self._connect()
        assert self.sock is not None
        try:
            self.sock.sendall(payload)
        except OSError:
            self._connect()
            assert self.sock is not None
            self.sock.sendall(payload)

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None


class GestureTcpServer:
    def __init__(self, host: str, port: int, timeout_s: float):
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.listen_sock: socket.socket | None = None
        self.client_sock: socket.socket | None = None

    def _ensure_listen(self) -> None:
        if self.listen_sock is not None:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.listen(1)
        sock.settimeout(self.timeout_s)
        self.listen_sock = sock

    def start(self) -> None:
        self._ensure_listen()

    def _ensure_client(self) -> None:
        if self.client_sock is not None:
            return
        self._ensure_listen()
        assert self.listen_sock is not None
        while self.client_sock is None:
            try:
                conn, addr = self.listen_sock.accept()
            except socket.timeout:
                continue
            conn.settimeout(self.timeout_s)
            self.client_sock = conn
            print(f"[smart-server] client connected: {addr[0]}:{addr[1]}", flush=True)

    def send_line(self, line: str) -> None:
        payload = (line.rstrip("\n") + "\n").encode("utf-8")
        if self.client_sock is None:
            print("[smart-server] waiting for client connection before send...", flush=True)
        self._ensure_client()
        assert self.client_sock is not None
        try:
            self.client_sock.sendall(payload)
        except OSError:
            try:
                self.client_sock.close()
            except OSError:
                pass
            self.client_sock = None
            self._ensure_client()
            assert self.client_sock is not None
            self.client_sock.sendall(payload)

    def close(self) -> None:
        if self.client_sock is not None:
            try:
                self.client_sock.close()
            except OSError:
                pass
            self.client_sock = None
        if self.listen_sock is not None:
            try:
                self.listen_sock.close()
            except OSError:
                pass
            self.listen_sock = None


def _send_tcp_line(transport, line: str) -> bool:
    try:
        transport.send_line(line)
        return True
    except ConnectionRefusedError:
        h = getattr(transport, "host", "?")
        p = getattr(transport, "port", "?")
        print(f"[smart] TCP rifiutato ({h}:{p}). Usa --dry-run o avvia il ricevitore.", flush=True)
        return False
    except OSError as e:
        if getattr(e, "errno", None) in (61, 111, 10061):
            h = getattr(transport, "host", "?")
            p = getattr(transport, "port", "?")
            print(f"[smart] TCP nessun peer su {h}:{p}. ({e!r})", flush=True)
            return False
        raise


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    # Audio
    p.add_argument("--segment-seconds", type=float, default=5.0)
    p.add_argument("--sample-rate", type=int, default=48000)
    p.add_argument("--mic-channels", type=int, choices=(1, 2), default=1)
    p.add_argument("--sd-device", type=int, default=None)
    p.add_argument("--save-dir", default="runtime_chunks_live/smart_mic")
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
                   help="Se p(music) >= questa soglia, forziamo MUSIC ignorando la VAD (la VAD "
                        "non riconosce la musica come speech e la scarterebbe come silenzio).")

    # Shazam (music path)
    p.add_argument("--index-path", default="shazam/index.pkl")
    p.add_argument(
        "--song-to-policy",
        default=(
            "salsa:2,salsa4:2,"
            "thriller:4,"
            "violin:3,"
            "67:5,sixseven:5,sei sette:5,"
            "gesti:1,"
            "stand:0"
        ),
    )
    p.add_argument("--min-confidence", type=float, default=0.0)
    p.add_argument("--min-votes", type=int, default=10)
    p.add_argument("--send-repeat-after", type=float, default=8.0)

    # OpenAI (speech path)
    p.add_argument("--openai-stt-model", default="gpt-4o-mini-transcribe")
    p.add_argument("--chat-model", default="gpt-4o-mini")
    p.add_argument(
        "--chat-system-prompt",
        default="Reply briefly and naturally in English.",
    )
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

    # Misc
    p.add_argument("--device", default=None, help="'cuda' o 'cpu'. Default: cuda se disponibile.")
    p.add_argument("--torch-dtype", default="float32", choices=("float32", "float16"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def format_line(template: str, **kwargs) -> str:
    safe_text = (kwargs.get("text") or "").replace("\n", " ").strip()
    kwargs["text"] = safe_text
    return template.format(**kwargs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
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
    id2l: dict | None = dict(getattr(cfg, "id2label", None) or {}) if cfg is not None else None

    print("[smart] loading Silero VAD ...", flush=True)
    vad_model, get_ts = load_silero_vad()

    matcher = LocalShazamMatcher.load_index(str(index_path))
    mapping = parse_song_map(args.song_to_policy)

    save_dir = Path(args.save_dir)
    if not save_dir.is_absolute():
        save_dir = (ROOT / save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    # TCP
    if args.tcp_mode == "server":
        transport: GestureTcpClient | GestureTcpServer = GestureTcpServer(
            host=str(args.server_host), port=int(args.server_port), timeout_s=float(args.connect_timeout),
        )
        transport.start()
        print(f"[smart-server] listening on {args.server_host}:{args.server_port}", flush=True)
    else:
        transport = GestureTcpClient(
            host=str(args.client_host), port=int(args.client_port), timeout_s=float(args.connect_timeout),
        )
        print(f"[smart-client] target={args.client_host}:{args.client_port}", flush=True)

    print(f"[smart] Shazam index: {index_path}", flush=True)
    print(
        f"[smart] segment={args.segment_seconds}s mic_sr={args.sample_rate} ast_sr={args.ast_sample_rate} "
        f"speech_margin={args.speech_margin} vad_silence_fraction={args.vad_silence_fraction}",
        flush=True,
    )

    last_policy_sent: int | None = None
    last_send_ts: float = 0.0
    ast_sr = int(args.ast_sample_rate)

    try:
        idx = 0
        while True:
            t0 = time.monotonic()
            wait_tts_idle(verbose=bool(args.verbose))
            raw = record_mono_f32(
                sample_rate=int(args.sample_rate),
                channels=int(args.mic_channels),
                duration_s=float(args.segment_seconds),
                sd_device=args.sd_device,
            )
            clip_path = save_dir / f"chunk_{idx:06d}.wav"
            try:
                write_wav_s16le_mono(clip_path, raw, int(args.sample_rate))
            except OSError as e:
                print(f"[smart] wav write err: {e!r}", flush=True)

            # 1) AST + VAD
            audio_ast = resample_f32(raw, int(args.sample_rate), ast_sr)
            ast_in = {"raw": audio_ast, "sampling_rate": ast_sr}
            try:
                out_list = clf(ast_in, top_k=int(args.ast_top_k))
            except TypeError:
                out_list = clf(ast_in)
            if not isinstance(out_list, list):
                out_list = [out_list]  # type: ignore[assignment]
            out_list = _expand_id2label(out_list, id2l)
            p_sp, p_mu = _label_scores_ast(out_list, args.verbose)

            vad_frac = silero_speech_fraction(
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

            # -------------------------------------------------------------
            # MUSIC branch: Shazam + POLICY_SWITCH
            # -------------------------------------------------------------
            if branch == "music":
                result = matcher.match(str(clip_path))
                policy_id = resolve_policy(result.song_path, result.song_id, mapping)
                song_name = Path(result.song_path).stem if result.song_path else "unknown"
                if (policy_id is None or float(result.confidence) < float(args.min_confidence)
                        or int(result.votes) < int(args.min_votes)):
                    if args.verbose:
                        print(
                            f"[smart] music skip song={song_name} conf={result.confidence:.2f} "
                            f"votes={result.votes} policy={policy_id}",
                            flush=True,
                        )
                else:
                    now = time.monotonic()
                    if (last_policy_sent is not None and policy_id == last_policy_sent
                            and (now - last_send_ts) < float(args.send_repeat_after)):
                        if args.verbose:
                            print(f"[smart] throttled policy={policy_id}", flush=True)
                    else:
                        line = f"[POLICY_SWITCH],{policy_id}"
                        if args.dry_run:
                            print(
                                f"[smart] DRY-RUN {line} song={song_name} conf={result.confidence:.2f} "
                                f"votes={result.votes}",
                                flush=True,
                            )
                            last_policy_sent = policy_id
                            last_send_ts = now
                        elif _send_tcp_line(transport, line):
                            print(
                                f"[smart] sent {line} song={song_name} conf={result.confidence:.2f} "
                                f"votes={result.votes}",
                                flush=True,
                            )
                            last_policy_sent = policy_id
                            last_send_ts = now

            # -------------------------------------------------------------
            # SPEECH branch: OpenAI + GESTI
            # -------------------------------------------------------------
            elif branch == "speech":
                try:
                    text, stt_s = transcribe_with_openai(clip_path, str(args.openai_stt_model))
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

                override_line, forced_reply = _special_routing_from_text(text)
                chat_api_seconds = 0.0
                ai_response_text = forced_reply or ""
                if forced_reply is None:
                    try:
                        ai_response_text, chat_api_seconds = generate_ai_response_text(
                            user_text=text or "silenzio",
                            model=str(args.chat_model),
                            system_prompt=str(args.chat_system_prompt),
                        )
                    except Exception as exc:  # noqa: BLE001
                        ai_response_text = "Scusa, c'e' stato un errore."
                        print(f"[smart] ai-reply error: {exc}", flush=True)

                # TTS (pre-calcolo durata, poi play dopo l'invio TCP)
                tts_seconds = 0.0
                tts_api_seconds = 0.0
                tts_audio: np.ndarray | None = None
                tts_sr = 0
                if not bool(args.no_speak_response):
                    spoken_text = str(args.speak_text_template).format(
                        text=text or "silenzio", ai_response=ai_response_text,
                    )
                    try:
                        tts_wav, tts_api_seconds = synthesize_tts_wav_with_openai(
                            spoken_text, model=str(args.tts_model), voice=str(args.tts_voice),
                        )
                        tts_audio, tts_sr = decode_wav_bytes(tts_wav)
                        tts_seconds = wav_duration_seconds_from_decoded(tts_audio, tts_sr)
                        if tts_seconds <= 0.0 or tts_seconds > 120.0:
                            words = len((spoken_text or "").split())
                            tts_seconds = max(0.8, words / 2.7)
                    except Exception as exc:  # noqa: BLE001
                        tts_audio = None
                        print(f"[smart] tts error: {exc}", flush=True)

                line = format_line(
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
                    if _send_tcp_line(transport, line):
                        print(f"[smart] sent line={line!r} text={preview!r}", flush=True)

                if tts_audio is not None and not bool(args.no_speak_response):
                    play_audio_array_background(tts_audio, tts_sr, speaker_device=args.speaker_device)

            if args.verbose:
                print(f"[smart] chunk wall time {time.monotonic() - t0:.2f}s", flush=True)
            idx += 1
    except KeyboardInterrupt:
        print("\n[smart] stopped", flush=True)
    finally:
        transport.close()


if __name__ == "__main__":
    main()
