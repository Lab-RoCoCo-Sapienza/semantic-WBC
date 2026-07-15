#!/usr/bin/env python3
"""Live microphone listener: Shazam match -> movement policy command -> TCP client.

Uses your PC’s microphone (room audio: music from speakers, etc.).

**macOS (default):** cattura con **PortAudio** (``sounddevice``), stesso schema di
``g1_unint`` (``G1RobotRemote``): niente demuxer AVFoundation / bug durata-livello.
Indice dispositivo: ``--sd-device N`` oppure lista::

    python3 -c "import sounddevice as sd; print(sd.query_devices())"

**macOS (alternativa ffmpeg):** ``--capture-backend ffmpeg`` + AVFoundation
``none:AUDIO_INDEX`` audio-only. Lista::

    ffmpeg -f avfoundation -list_devices true -i ''

Override default ffmpeg con env ``ROBO_SHAZAM_MIC``. Override backend:
``ROBO_SHAZAM_CAPTURE=sounddevice|ffmpeg``.

I segmenti microfono sono **WAV temporanei** (default: nessun file in ``--watch-dir``;
solo tempfile per Shazam, poi cancellati). Con ``--save-wav`` scrivi in ``--watch-dir``
come ``<prefisso>_NNNNNN.wav`` (default 48000 Hz, vedi ``--sample-rate``). Con
``--save-wav``, restano dopo l’analisi salvo ``--delete-chunks``; all’avvio la
cartella viene svuotata salvo ``--no-clean``.

**Qualità WAV su macOS (riferimenti comuni):** AVFoundation spesso dà **PCM float32**
(``pcm_f32le``); conviene **ricalcampare e forzare s16** con filtri (vedi
``aresample`` + ``aformat`` in questo script). In Audio MIDI Setup, se il microfono
è a **24 bit**, alcuni utenti segnalano artefatti finché non impostano **16 bit** per
il dispositivo di ingresso. Convenzione dispositivi:
https://apple.stackexchange.com/questions/326388/terminal-command-to-record-audio-through-macbook-microphone
— discussione su ``ffmpeg -f avfoundation -i ":1"`` (solo audio). Opzione
``--wav-capture timed`` usa **un ffmpeg con ``-t`` per file** (come nelle guide), spesso
più stabile del muxer ``segment`` per i WAV.

Sotto ``--skip-if-peak-below-dbfs`` (default -48 dBFS) non parte Shazam né l’invio TCP:
stanza silenziosa o ingresso troppo debole → niente comandi al sim/robot.

**Perché a volte “indovina” male:** il microfono non è il player MP3: cattura **la stanza**
(rumore, riverbero, notifiche, altra musica lontana). Con **pochi ``votes``** (1–3) il
fingerprint locale spesso è un falso positivo. Prova ``--segment-seconds 10``,
``--min-confidence 2``, ``--strict`` (alza ancora votes/confidence), e ``-v`` per i dettagli.

Example (solo riconoscimento, niente TCP; macOS = sounddevice di default)::

    python scripts/listen_shazam_and_send.py --dry-run

Forzare ffmpeg + AVFoundation (come prima)::

    python scripts/listen_shazam_and_send.py --dry-run --capture-backend ffmpeg --mic-input none:1

WAV solo via muxer ffmpeg ``segment`` (Linux / debug)::

    python scripts/listen_shazam_and_send.py --dry-run --capture-backend ffmpeg --wav-capture segment

Example con invio al robot / sim::

    python scripts/listen_shazam_and_send.py \
      --song-to-policy "dynamite:3,swim:5,salsa4:9,salsa:7,thriller:6,gdance:8,bts:2" \
      --client-host 127.0.0.1 --client-port 8765 \
      --segment-seconds 5
"""

from __future__ import annotations

import argparse
import audioop
import json
import os
import queue
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _temp_clip_path(*, prefix: str) -> Path:
    fd, name = tempfile.mkstemp(prefix=f"{prefix}_", suffix=".wav")
    os.close(fd)
    return Path(name)


def _clip_path_for_idx(
    *, save_wav: bool, watch_dir: Path, chunk_prefix: str, idx: int
) -> Path:
    if save_wav:
        return watch_dir / f"{chunk_prefix}_{idx:06d}.wav"
    return _temp_clip_path(prefix=chunk_prefix)


def parse_song_map(value: str) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    if not value.strip():
        return out
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid map entry '{item}'. Expected key:policy_id")
        key, pid = item.split(":", 1)
        out.append((key.strip().lower(), int(pid)))
    return out


def _fmt_match_dbg(result, verbose: bool) -> str:
    if not verbose:
        return ""
    off = result.offset_sec
    off_s = f"{off:.2f}s" if off is not None else "n/a"
    return (
        f" | strat={result.strategy} offset={off_s} "
        f"hits={int(result.total_hits)} votes={int(result.votes)}"
    )


def resolve_policy(song_path: str | None, song_id: str | None, mapping: list[tuple[str, int]]) -> int | None:
    if not mapping:
        return None
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


def _safe_chunk_prefix(raw: str) -> str:
    s = (raw or "mic").strip() or "mic"
    out = "".join(c if c.isalnum() or c in "-_" else "_" for c in s)
    return (out[:48] or "mic").strip("_") or "mic"


def _resolve_mic_input(raw: str) -> str:
    """Resolve empty --mic-input to a safe platform default."""
    if raw.strip():
        return raw.strip()
    if sys.platform == "darwin":
        # Often [0]=virtual (BlackHole), [1]=built-in mic — ":0" wrongly picks audio 0.
        return os.environ.get("ROBO_SHAZAM_MIC", "none:1").strip() or "none:1"
    return raw


def _offline_ffmpeg_af_chain(
    *,
    preset: str,
    channels: int,
    gain_db: float,
    auto_normalize: bool,
) -> str:
    """Filtri su PCM gia` a ``--sample-rate`` (nessun aresample: evita bug AVFoundation)."""
    ch = 2 if int(channels) == 2 else 1
    layout = "stereo" if ch == 2 else "mono"
    parts: list[str] = []
    if preset == "room":
        parts.append("highpass=f=100")
    if auto_normalize:
        parts.append("dynaudnorm=f=500:g=3:p=0.95:m=30")
    if abs(gain_db) >= 0.01:
        parts.append(f"volume={gain_db:+.2f}dB")
    parts.append(f"aformat=sample_fmts=s16:channel_layouts={layout}")
    return ",".join(parts)


def _ffmpeg_process_s16le_pcm(
    pcm: bytes,
    *,
    sample_rate: int,
    channels: int,
    af: str,
) -> bytes:
    """Applica ``-af`` a PCM s16le interleaved (stdin -> stdout)."""
    if not pcm:
        return pcm
    ch = int(channels)
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-f",
            "s16le",
            "-ar",
            str(int(sample_rate)),
            "-ac",
            str(ch),
            "-i",
            "pipe:0",
            "-af",
            af,
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "pipe:1",
        ],
        input=pcm,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        err = (proc.stderr or b"").decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"ffmpeg filter PCM failed (exit {proc.returncode}): {err}")
    return proc.stdout


def _write_wav_s16le(path: Path, pcm: bytes, *, sample_rate: int, channels: int) -> None:
    ch = int(channels)
    if ch not in (1, 2):
        raise ValueError("channels must be 1 or 2")
    nframes = len(pcm) // (2 * ch)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm[: nframes * 2 * ch])


class _SounddeviceMicStream:
    """Stesso schema di ``g1_unint/src/robot_interfaces/g1_robot_remote.py``: PortAudio
    (CoreAudio su macOS), float32 in callback, downmix, ``audioop.ratecv`` verso il rate
    di uscita. Nessun demuxer AVFoundation → niente PTS / durata dimezzata.
    """

    def __init__(
        self,
        *,
        sd_device: int | None,
        out_rate: int,
        out_channels: int,
        capture_rate: int | None,
        capture_channels: int | None,
        blocksize: int = 1024,
    ) -> None:
        import numpy as np
        import sounddevice as sd

        self._np = np
        self._sd = sd
        dev = sd.query_devices(sd_device, "input") if sd_device is not None else sd.query_devices(None, "input")
        self._capture_rate = int(
            capture_rate
            or float(dev.get("default_samplerate") or 48000)
        )
        max_in = int(dev.get("max_input_channels") or 1)
        cap_ch = int(capture_channels or min(2 if int(out_channels) == 2 else 1, max(1, max_in)))
        self._capture_channels = max(1, cap_ch)
        self._out_rate = int(out_rate)
        self._out_ch = max(1, min(2, int(out_channels)))
        self._device = sd_device
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=400)
        self._ratecv_state: object | None = None

        def _cb(indata, frames, _time_info, status) -> None:  # noqa: ANN001
            if status:
                return
            try:
                np = self._np
                x = np.asarray(indata, dtype=np.float32)
                if x.ndim == 1:
                    x = x.reshape(-1, 1)
                x = np.clip(x, -1.0, 1.0)
                if self._out_ch == 1:
                    if x.shape[1] > 1:
                        xm = np.mean(x, axis=1)
                    else:
                        xm = x[:, 0]
                    mono16 = (xm * 32767.0).astype(np.int16).tobytes()
                    raw = mono16
                else:
                    if x.shape[1] == 1:
                        lr = np.stack([x[:, 0], x[:, 0]], axis=1)
                    else:
                        lr = x[:, :2]
                    raw = (lr * 32767.0).astype(np.int16).tobytes()
                if self._capture_rate != self._out_rate:
                    raw, self._ratecv_state = audioop.ratecv(
                        raw,
                        2,
                        self._out_ch,
                        self._capture_rate,
                        self._out_rate,
                        self._ratecv_state,
                    )
                self._queue.put_nowait(raw)
            except queue.Full:
                pass
            except Exception:
                return

        self._stream = sd.InputStream(
            samplerate=self._capture_rate,
            channels=self._capture_channels,
            dtype="float32",
            device=sd_device,
            callback=_cb,
            blocksize=int(blocksize),
        )
        self._stream.start()

    def close(self) -> None:
        try:
            self._stream.stop()
        except Exception:
            pass
        try:
            self._stream.close()
        except Exception:
            pass

    def read_s16le(self, duration_s: float) -> bytes:
        target = int(self._out_rate * duration_s) * 2 * self._out_ch
        out = bytearray()
        while len(out) < target:
            out.extend(self._queue.get())
        return bytes(out[:target])


def _default_capture_backend() -> str:
    if os.environ.get("ROBO_SHAZAM_CAPTURE", "").strip().lower() in ("ffmpeg", "sounddevice", "sd"):
        v = os.environ.get("ROBO_SHAZAM_CAPTURE", "").strip().lower()
        return "sounddevice" if v == "sd" else v
    if sys.platform == "darwin":
        return "sounddevice"
    return "ffmpeg"


def _audio_filter_chain(
    sample_rate: int,
    *,
    preset: str,
    channels: int,
    gain_db: float = 0.0,
    auto_normalize: bool = True,
) -> str:
    """Normalize device audio (often float32 from AVFoundation) to PCM s16 @ sr.

    FIX CRITICO macOS #1 (durata): il demuxer AVFoundation nega il rate nativo del
    device (tipico MacBook Pro: 96 kHz) mentre ffmpeg timestampa i sample con un
    rate diverso → produce WAV che durano la META` di quello chiesto (5s -> 2.2s).
    ``aresample=async=1000:first_pts=0`` compensa lo scivolamento PTS.

    FIX CRITICO macOS #2 (livello): con Input Volume basso o Voice Isolation attiva,
    il WAV esce a ~-30 dBFS peak (-50 dBFS RMS) → troppo basso per Shazam (soglia
    utile ~-20 dBFS peak). ``dynaudnorm`` porta il livello a un target stabile
    compressandolo moltissimo; poi ``volume`` applica un trim extra se chiesto.
    """
    sr = int(sample_rate)
    ch = 2 if int(channels) == 2 else 1
    layout = "stereo" if ch == 2 else "mono"
    rs = f"aresample=async=1000:first_pts=0:osr={sr}:filter_size=96:cutoff=0.97"
    if preset == "room":
        front = "highpass=f=100,"
    else:
        front = ""
    parts = [front, rs]
    if auto_normalize:
        # f=500ms frame / g=3 light smoothing / p=0.95 peak / m=30dB max gain.
        # Con Input Volume basso o Voice Isolation attiva questi parametri portano
        # peak da ~-29 dBFS a ~-7 dBFS senza pompare troppo sulla musica (verificato
        # su MacBook Pro M-series). Parametri più "mansueti" (m=20) lasciano il mic
        # sotto la soglia utile di Shazam (-20 dBFS).
        parts.append(",dynaudnorm=f=500:g=3:p=0.95:m=30")
    if abs(gain_db) >= 0.01:
        parts.append(f",volume={gain_db:+.2f}dB")
    parts.append(f",aformat=sample_fmts=s16:channel_layouts={layout}")
    return "".join(parts)


def _ffmpeg_mic_cmd(
    out_pattern: str,
    segment_s: float,
    input_name: str,
    *,
    sample_rate: int,
    no_audio_filter: bool,
    mic_preset: str,
    mic_channels: int,
    gain_db: float = 0.0,
    auto_normalize: bool = True,
) -> list[str]:
    """Capture mic -> segmented WAV (PCM s16le).

    Uses ``-segment_format wav`` and omits ``-reset_timestamps 1`` (that combo often
    yields broken duration / chipmunk playback for short WAV segments).
    """
    base = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-probesize",
        "2000000",
        "-analyzeduration",
        "4000000",
        "-thread_queue_size",
        "1024",
    ]
    if sys.platform == "darwin":
        src = _resolve_mic_input(input_name)
        # -use_wallclock_as_timestamps 1 corregge il PTS di AVFoundation senza bisogno
        # del hack -segment_atclocktime 1 (che su input audio-only produceva segmenti
        # anomali ~2.2 s invece di --segment-seconds).
        input_args = [
            "-use_wallclock_as_timestamps",
            "1",
            "-f",
            "avfoundation",
            "-i",
            src,
        ]
    elif sys.platform.startswith("linux"):
        src = input_name or "default"
        input_args = ["-f", "alsa", "-i", src]
    else:
        # Windows fallback; user may override input via --mic-input.
        src = input_name or "audio=Microphone"
        input_args = ["-f", "dshow", "-i", src]
    sr = int(sample_rate)
    ch = 2 if int(mic_channels) == 2 else 1
    seg_args: list[str] = []
    if not no_audio_filter:
        seg_args.extend(
            [
                "-af",
                _audio_filter_chain(
                    sr,
                    preset=mic_preset,
                    channels=ch,
                    gain_db=gain_db,
                    auto_normalize=auto_normalize,
                ),
            ]
        )
    else:
        seg_args.extend(["-sample_fmt", "s16"])
    seg_args.extend(
        [
            "-ac",
            str(ch),
            "-ar",
            str(sr),
            "-c:a",
            "pcm_s16le",
            "-f",
            "segment",
            "-segment_time",
            str(segment_s),
            # -reset_timestamps 1 è richiesto per chunk WAV consistenti con input live.
            "-reset_timestamps",
            "1",
            "-segment_format",
            "wav",
            "-write_empty_segments",
            "0",
            out_pattern,
        ]
    )
    return base + input_args + seg_args


def _ffmpeg_timed_mic_wav(
    out_path: Path,
    duration_s: float,
    input_name: str,
    *,
    sample_rate: int,
    no_audio_filter: bool,
    mic_preset: str,
    mic_channels: int,
    gain_db: float = 0.0,
    auto_normalize: bool = True,
) -> list[str]:
    """One ffmpeg run = one WAV of exactly ``duration_s`` (classic macOS recipe: ``-t``)."""
    base = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-probesize",
        "2000000",
        "-analyzeduration",
        "4000000",
        "-thread_queue_size",
        "1024",
    ]
    if sys.platform == "darwin":
        input_args = [
            "-use_wallclock_as_timestamps",
            "1",
            "-f",
            "avfoundation",
            "-i",
            _resolve_mic_input(input_name),
        ]
    elif sys.platform.startswith("linux"):
        src = input_name or "default"
        input_args = ["-f", "alsa", "-i", src]
    else:
        src = input_name or "audio=Microphone"
        input_args = ["-f", "dshow", "-i", src]
    sr = int(sample_rate)
    ch = 2 if int(mic_channels) == 2 else 1
    mid: list[str] = ["-t", str(float(duration_s))]
    if not no_audio_filter:
        mid.extend(
            [
                "-af",
                _audio_filter_chain(
                    sr,
                    preset=mic_preset,
                    channels=ch,
                    gain_db=gain_db,
                    auto_normalize=auto_normalize,
                ),
            ]
        )
    else:
        mid.extend(["-sample_fmt", "s16"])
    mid.extend(
        [
            "-ac",
            str(ch),
            "-ar",
            str(sr),
            "-c:a",
            "pcm_s16le",
            str(out_path),
        ]
    )
    return base + input_args + mid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index-path", default="shazam/index.pkl")
    p.add_argument("--watch-dir", default="runtime_chunks_live")
    p.add_argument(
        "--save-wav",
        action="store_true",
        help="Salva i WAV del microfono sotto --watch-dir. Default: no (tempfile solo per "
        "Shazam, ideale per sim / disco pulito).",
    )
    p.add_argument(
        "--chunk-prefix",
        default="mic",
        help="Prefisso nomi clip: <prefisso>_000000.wav (solo microfono, non il bus audio).",
    )
    p.add_argument(
        "--segment-seconds",
        type=float,
        default=5.0,
        help="Durata nominale di ogni clip (secondi). In modalità segment usa anche il clock di sistema.",
    )
    p.add_argument(
        "--sample-rate",
        type=int,
        default=48000,
        help="Sample rate output WAV (default 48000, tipico CoreAudio; prova 44100 se suona strano).",
    )
    p.add_argument("--poll-interval", type=float, default=0.1)
    p.add_argument(
        "--stability-wait",
        type=float,
        default=0.35,
        help="Secondi di attesa stabilità dimensione file prima di analizzare (segment WAV).",
    )
    p.add_argument(
        "--wav-capture",
        choices=("segment", "timed"),
        default="timed",
        help="timed (default): un ffmpeg -t per WAV, WAV sempre chiusi correttamente. "
        "segment: muxer segment multi-file (più veloce ma può lasciare l'ultimo file corrotto).",
    )
    p.add_argument(
        "--no-audio-filter",
        action="store_true",
        help="Disabilita aresample+aformat (solo confronto / debug).",
    )
    p.add_argument(
        "--gain-boost-db",
        type=float,
        default=0.0,
        help="dB di guadagno extra applicati DOPO dynaudnorm (es. 6 o 12). Utile "
        "quando il mic interno parte a -30 dBFS. Negativo = taglia.",
    )
    p.add_argument(
        "--no-auto-normalize",
        action="store_true",
        help="Disattiva dynaudnorm (normalizzazione dinamica). Per default è ON: porta "
        "il livello a un target stabile per Shazam anche con Input Volume basso.",
    )
    p.add_argument(
        "--mic-preset",
        choices=("minimal", "room"),
        default="minimal",
        help="minimal (default): solo conversione formato, timbro fedele al mic. "
        "room: aggiunge highpass=100Hz per togliere ronzio ma rende l'audio 'telefonico'.",
    )
    p.add_argument(
        "--mic-channels",
        type=int,
        choices=(1, 2),
        default=1,
        help="2=stereo (se il device lo dà; spesso più “aria” con casse lontane); 1=mono.",
    )
    p.add_argument(
        "--capture-backend",
        choices=("auto", "ffmpeg", "sounddevice"),
        default="auto",
        help="auto: su macOS sounddevice (come g1_unint), altrove ffmpeg. "
        "sounddevice: PortAudio/CoreAudio, ignora --mic-input. "
        "ffmpeg: ALSA/dshow/avfoundation + --mic-input.",
    )
    p.add_argument(
        "--sd-device",
        type=int,
        default=None,
        help="Indice ingresso sounddevice (None = default sistema). Ignorato con --capture-backend ffmpeg.",
    )
    p.add_argument(
        "--sd-blocksize",
        type=int,
        default=1024,
        help="Blocksize callback sounddevice (default 1024).",
    )
    p.add_argument(
        "--sd-capture-rate",
        type=int,
        default=0,
        help="Sample rate cattura device (0 = default del device, es. 48000/96000).",
    )
    p.add_argument(
        "--sd-capture-channels",
        type=int,
        default=0,
        help="Canali cattura (0 = min(2, max_input) come g1_unint; 1 o 2 espliciti).",
    )
    p.add_argument(
        "--mic-input",
        default="",
        help="Solo ffmpeg: dispositivo input (vuoto macOS -> ROBO_SHAZAM_MIC o 'none:1'). "
        "Con sounddevice usa --sd-device.",
    )
    p.add_argument("--song-to-policy", default="dynamite:3,swim:5,salsa4:9,salsa:7,thriller:6,gdance:8,bts:2")
    p.add_argument("--min-confidence", type=float, default=0.0)
    p.add_argument(
        "--min-votes",
        type=int,
        default=10,
        help="Voti minimi Shazam per accettare un match (default 10, meno falsi positivi).",
    )
    p.add_argument(
        "--skip-if-peak-below-dbfs",
        type=float,
        default=-48.0,
        help="Se il picco del clip (dBFS) è sotto questa soglia, non eseguire Shazam e non inviare "
        "TCP (stanza silenziosa / mic coperto). sounddevice: misura sul PCM grezzo pre-filtro. "
        "ffmpeg: sul WAV scritto. Disattiva effettivamente con -120.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Soglie più alte contro i falsi positivi: "
        "min_votes = max(attuale, 12), min_confidence = max(attuale, 2.0).",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log extra (strategia matcher, offset, total_hits).",
    )
    p.add_argument("--send-repeat-after", type=float, default=8.0)
    p.add_argument("--client-host", default="127.0.0.1")
    p.add_argument("--client-port", type=int, default=8765)
    p.add_argument("--connect-timeout", type=float, default=1.0)
    p.add_argument(
        "--dry-run",
        "--no-send",
        action="store_true",
        dest="dry_run",
        help="Solo cattura + Shazam + log (nessuna connessione TCP al client).",
    )
    p.add_argument(
        "--delete-chunks",
        action="store_true",
        help="Solo con --save-wav: dopo l'analisi elimina ogni clip dalla watch-dir.",
    )
    p.add_argument(
        "--keep-chunks",
        action="store_true",
        help="Deprecato: con --save-wav i file restano salvo --delete-chunks.",
    )
    p.add_argument("--no-clean", action="store_true")
    return p.parse_args()


def _s16le_peak_dbfs(pcm: bytes, channels: int) -> float:
    """Picco normalizzato dBFS su PCM s16le interleaved (pre-filtro, es. sounddevice)."""
    import math

    ch = max(1, int(channels))
    if not pcm or len(pcm) < 2 * ch:
        return -120.0
    max_abs = 0
    for off in range(0, len(pcm) - 1, 2):
        v = int.from_bytes(pcm[off : off + 2], "little", signed=True)
        a = abs(v)
        if a > max_abs:
            max_abs = a
    return float(20.0 * math.log10(max(max_abs, 1e-9) / 32768.0))


def _measure_peak_dbfs(path: Path) -> float | None:
    """Returns peak level in dBFS (e.g. -3.0) or None if not measurable."""
    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostats",
                "-loglevel",
                "info",
                "-i",
                str(path),
                "-af",
                "volumedetect",
                "-f",
                "null",
                "/dev/null" if sys.platform != "win32" else "NUL",
            ],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in proc.stderr.splitlines():
        if "max_volume:" in line:
            try:
                return float(line.split("max_volume:")[1].strip().split()[0])
            except (ValueError, IndexError):
                return None
    return None


def _file_stable(path: Path, wait_s: float) -> bool:
    try:
        s1 = path.stat().st_size
    except OSError:
        return False
    if s1 <= 0:
        return False
    time.sleep(max(0.0, wait_s))
    try:
        s2 = path.stat().st_size
    except OSError:
        return False
    return s1 == s2


def _send_command(host: str, port: int, timeout_s: float, payload: dict) -> None:
    msg = (json.dumps(payload, ensure_ascii=True) + "\n").encode("utf-8")
    with socket.create_connection((host, port), timeout=timeout_s) as sock:
        sock.sendall(msg)


def _process_one_clip(
    clip: Path,
    *,
    matcher,
    mapping: list[tuple[str, int]],
    args: argparse.Namespace,
    remove_after_match: bool,
    last_policy_sent: int | None,
    last_send_ts: float,
) -> tuple[int | None, float]:
    """Run Shazam + optional TCP send for one finished clip; optionally delete file."""
    if args.verbose:
        peak = _measure_peak_dbfs(clip)
        if peak is not None:
            tag = ""
            if peak <= -25.0:
                tag = "  <- TROPPO BASSO (alza Input Volume / togli Voice Isolation / --gain-boost-db 10)"
            elif peak <= -15.0:
                tag = "  <- ok-ish"
            else:
                tag = "  <- buono"
            print(f"[listener] {clip.name} peak={peak:+.1f} dBFS{tag}", flush=True)
    result = matcher.match(str(clip))
    song_name = Path(result.song_path).stem if result.song_path else "unknown"
    policy_id = resolve_policy(result.song_path, result.song_id, mapping)
    if (
        policy_id is None
        or float(result.confidence) < args.min_confidence
        or int(result.votes) < args.min_votes
    ):
        print(
            f"[listener] {clip.name} -> skip "
            f"(song={song_name}, conf={result.confidence:.2f}, votes={result.votes}, "
            f"policy={policy_id}, strat={result.strategy})"
            f"{_fmt_match_dbg(result, args.verbose)}",
            flush=True,
        )
        if remove_after_match:
            try:
                clip.unlink()
            except OSError:
                pass
        return last_policy_sent, last_send_ts

    now = time.monotonic()
    if (
        last_policy_sent is not None
        and policy_id == last_policy_sent
        and (now - last_send_ts) < args.send_repeat_after
    ):
        print(f"[listener] {clip.name} -> same policy {policy_id}, throttled", flush=True)
        if remove_after_match:
            try:
                clip.unlink()
            except OSError:
                pass
        return last_policy_sent, last_send_ts

    cmd = f"[POLICY_SWITCH],{policy_id}"
    payload = {
        "command": cmd,
        "policy_id": policy_id,
        "song_path": result.song_path,
        "song_id": result.song_id,
        "confidence": float(result.confidence),
        "votes": int(result.votes),
        "offset_sec": result.offset_sec,
        "clip": clip.name,
    }
    if args.dry_run:
        last_policy_sent = policy_id
        last_send_ts = now
        print(
            f"[listener] {clip.name} -> DRY-RUN {cmd} "
            f"(song={song_name}, conf={result.confidence:.2f}, votes={result.votes})"
            f"{_fmt_match_dbg(result, args.verbose)}",
            flush=True,
        )
    else:
        try:
            _send_command(args.client_host, args.client_port, args.connect_timeout, payload)
            last_policy_sent = policy_id
            last_send_ts = now
            print(
                f"[listener] {clip.name} -> {cmd} "
                f"(song={song_name}, conf={result.confidence:.2f}, votes={result.votes})"
                f"{_fmt_match_dbg(result, args.verbose)}",
                flush=True,
            )
        except OSError as e:
            print(f"[listener] send failed for {cmd}: {e}", flush=True)

    if remove_after_match:
        try:
            clip.unlink()
        except OSError:
            pass
    return last_policy_sent, last_send_ts


def main() -> None:
    args = parse_args()
    if args.strict:
        args.min_votes = max(int(args.min_votes), 12)
        args.min_confidence = max(float(args.min_confidence), 2.0)
        print(
            f"[listener] --strict: min_votes={args.min_votes}, min_confidence={args.min_confidence}",
            flush=True,
        )

    save_wav = bool(args.save_wav)
    remove_after_match = bool(args.delete_chunks) and save_wav
    if getattr(args, "keep_chunks", False) and remove_after_match:
        raise SystemExit("Non usare insieme --keep-chunks e --delete-chunks.")
    if bool(args.delete_chunks) and not save_wav:
        print(
            "[listener] nota: --delete-chunks non ha effetto senza --save-wav "
            "(le clip sono già temporanee).",
            flush=True,
        )

    sys.path.insert(0, str((ROOT / "shazam").resolve()))
    from local_fingerprint import LocalShazamMatcher  # noqa: E402

    mapping = parse_song_map(args.song_to_policy)
    index_path = Path(args.index_path).expanduser()
    if not index_path.is_absolute():
        index_path = (ROOT / index_path).resolve()
    if not index_path.is_file():
        raise SystemExit(f"Shazam index not found: {index_path}")

    watch_dir = Path(args.watch_dir).expanduser()
    if not watch_dir.is_absolute():
        watch_dir = (ROOT / watch_dir).resolve()
    watch_dir.mkdir(parents=True, exist_ok=True)

    if save_wav and not args.no_clean:
        for old in watch_dir.glob("*"):
            if old.is_file() and old.suffix.lower() in (".mp3", ".wav", ".flac", ".m4a"):
                try:
                    old.unlink()
                except OSError:
                    pass

    matcher = LocalShazamMatcher.load_index(str(index_path))
    print(f"[listener] index loaded: {index_path}", flush=True)
    if save_wav:
        print(f"[listener] watch dir (persistenza WAV): {watch_dir}", flush=True)
    else:
        print(
            "[listener] WAV microfono: solo tempfile (nessun salvataggio in watch-dir); "
            "`--save-wav` per scrivere sotto --watch-dir.",
            flush=True,
        )
    chunk_prefix = _safe_chunk_prefix(args.chunk_prefix)
    segment_pattern = f"{chunk_prefix}_%06d.wav"
    _ch = int(args.mic_channels)
    print(
        "[listener] I .wav sono SOLO registrazioni MICROFONO (segmenti di "
        f"{args.segment_seconds:g}s, {_ch} ch @ {int(args.sample_rate)} Hz, preset={args.mic_preset!r}) "
        f"→ `{segment_pattern}`. NON è il bus audio del player.",
        flush=True,
    )
    print(
        "[listener] limite fisico: il microfono prende stanza + casse, non il bus audio del player — "
        "non ci sono magie ffmpeg che lo facciano suonare come uno stream pulito. "
        "Per qualita vicina a 'quello che esce dal Mac' serve un loopback (es. BlackHole) come ingresso, "
        "non il microfono integrato.",
        flush=True,
    )
    if save_wav:
        if remove_after_match:
            print(
                "[listener] clip: salvate sotto watch-dir, eliminate dopo analisi (--delete-chunks).",
                flush=True,
            )
        else:
            print(
                "[listener] clip: salvate in watch-dir (nessuna --delete-chunks). "
                "Per non cancellare i file precedenti all'avvio, usa anche --no-clean.",
                flush=True,
            )
    else:
        print("[listener] clip: non salvate su disco (solo Shazam + eventuale TCP).", flush=True)
    if args.dry_run:
        print("[listener] dry-run: nessun invio TCP (solo log)", flush=True)
    else:
        print(f"[listener] sending to: {args.client_host}:{args.client_port}", flush=True)

    backend = str(args.capture_backend).strip().lower()
    if backend == "auto":
        backend = _default_capture_backend()
    if backend == "sounddevice":
        try:
            import sounddevice as _sd_check  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "Manca il pacchetto 'sounddevice' (stesso stack di g1_unint). "
                "Installa: pip install sounddevice   "
                "Oppure: --capture-backend ffmpeg"
            ) from exc

    mic_resolved = _resolve_mic_input(args.mic_input)
    if sys.platform == "darwin":
        print(
            "[listener] nota macOS: se l'audio suona 'strozzato' o la voce svanisce, "
            "apri Control Center -> Mic Mode e imposta 'Standard' (disattiva Voice "
            "Isolation / Wide Spectrum). Queste modalita' sopprimono la musica e "
            "rendono Shazam cieco.",
            flush=True,
        )
    if backend == "sounddevice":
        import sounddevice as sd

        dev = sd.query_devices(args.sd_device, "input") if args.sd_device is not None else sd.query_devices(
            None, "input"
        )
        print(
            f"[listener] sounddevice ingresso: {dev['name']!r} index={dev['index']} "
            f"(override: --sd-device N; lista: python3 -c \"import sounddevice as sd; print(sd.query_devices())\")",
            flush=True,
        )
    elif sys.platform == "darwin" and backend == "ffmpeg":
        print(f"[listener] macOS ffmpeg AVFoundation input: {mic_resolved!r}", flush=True)
        if mic_resolved.startswith("none:"):
            print(
                "[listener] tip: if Shazam stays at votes=0, run "
                "`ffmpeg -f avfoundation -list_devices true -i ''` and set "
                "`--mic-input none:<audio_index>` or `ROBO_SHAZAM_MIC`.",
                flush=True,
            )

    no_af = bool(args.no_audio_filter)
    print(
        f"[listener] capture-backend={backend!r} wav-capture={args.wav_capture!r} "
        f"mic_preset={args.mic_preset!r} mic_channels={int(args.mic_channels)} "
        f"audio_filter={'off' if no_af else 'on'}",
        flush=True,
    )
    if backend == "ffmpeg" and args.wav_capture == "timed" and not no_af:
        print(
            "[listener] modalità ffmpeg timed: un processo ffmpeg per ogni clip (-t).",
            flush=True,
        )
    if backend == "sounddevice" and args.wav_capture == "segment":
        print(
            "[listener] nota: con sounddevice ``--wav-capture segment`` non è supportato; "
            "registro clip sequenziali in RAM (come timed).",
            flush=True,
        )

    ffmpeg: subprocess.Popen | None = None
    sd_mic: _SounddeviceMicStream | None = None
    scrap_segment_dir: Path | None = None
    last_policy_sent: int | None = None
    last_send_ts: float = 0.0

    try:
        if backend == "sounddevice":
            cap_rate = int(args.sd_capture_rate) if int(args.sd_capture_rate) > 0 else None
            cap_ch = int(args.sd_capture_channels) if int(args.sd_capture_channels) > 0 else None
            sd_mic = _SounddeviceMicStream(
                sd_device=args.sd_device,
                out_rate=int(args.sample_rate),
                out_channels=int(args.mic_channels),
                capture_rate=cap_rate,
                capture_channels=cap_ch,
                blocksize=int(args.sd_blocksize),
            )
            print(
                f"[listener] sounddevice: capture {sd_mic._capture_rate} Hz x{sd_mic._capture_channels} "
                f"→ WAV {int(args.sample_rate)} Hz x{int(args.mic_channels)} (schema g1_unint)",
                flush=True,
            )
            record_idx = 0
            while True:
                clip = _clip_path_for_idx(
                    save_wav=save_wav,
                    watch_dir=watch_dir,
                    chunk_prefix=chunk_prefix,
                    idx=record_idx,
                )
                print(f"[listener] sd record {record_idx} -> {clip.name} …", flush=True)
                try:
                    raw = sd_mic.read_s16le(float(args.segment_seconds))
                    thr = float(args.skip_if_peak_below_dbfs)
                    if thr > -119.0:
                        pk_in = _s16le_peak_dbfs(raw, int(args.mic_channels))
                        if pk_in < thr:
                            print(
                                f"[listener] sd record {record_idx} -> skip silenzio "
                                f"(picco ingresso {pk_in:.1f} dBFS < {thr:.0f} dBFS), niente Shazam/TCP",
                                flush=True,
                            )
                            if not save_wav and clip.exists():
                                try:
                                    clip.unlink()
                                except OSError:
                                    pass
                            record_idx += 1
                            continue
                    if no_af:
                        pcm = raw
                    else:
                        af = _offline_ffmpeg_af_chain(
                            preset=str(args.mic_preset),
                            channels=int(args.mic_channels),
                            gain_db=float(args.gain_boost_db),
                            auto_normalize=not bool(args.no_auto_normalize),
                        )
                        pcm = _ffmpeg_process_s16le_pcm(
                            raw,
                            sample_rate=int(args.sample_rate),
                            channels=int(args.mic_channels),
                            af=af,
                        )
                    _write_wav_s16le(
                        clip,
                        pcm,
                        sample_rate=int(args.sample_rate),
                        channels=int(args.mic_channels),
                    )
                except (RuntimeError, OSError) as exc:
                    print(f"[listener] cattura/filtro fallito: {exc}; riprovo tra 1s.", flush=True)
                    if not save_wav and clip.exists():
                        try:
                            clip.unlink()
                        except OSError:
                            pass
                    time.sleep(1.0)
                    continue
                try:
                    last_policy_sent, last_send_ts = _process_one_clip(
                        clip,
                        matcher=matcher,
                        mapping=mapping,
                        args=args,
                        remove_after_match=remove_after_match,
                        last_policy_sent=last_policy_sent,
                        last_send_ts=last_send_ts,
                    )
                finally:
                    if not save_wav and clip.exists():
                        try:
                            clip.unlink()
                        except OSError:
                            pass
                record_idx += 1

        elif args.wav_capture == "timed":
            record_idx = 0
            while True:
                clip = _clip_path_for_idx(
                    save_wav=save_wav,
                    watch_dir=watch_dir,
                    chunk_prefix=chunk_prefix,
                    idx=record_idx,
                )
                rec_cmd = _ffmpeg_timed_mic_wav(
                    clip,
                    args.segment_seconds,
                    args.mic_input,
                    sample_rate=int(args.sample_rate),
                    no_audio_filter=no_af,
                    mic_preset=str(args.mic_preset),
                    mic_channels=int(args.mic_channels),
                    gain_db=float(args.gain_boost_db),
                    auto_normalize=not bool(args.no_auto_normalize),
                )
                print(f"[listener] timed record {record_idx} -> {clip.name} …", flush=True)
                rc = subprocess.run(rec_cmd, check=False).returncode
                if rc != 0:
                    print(
                        f"[listener] ffmpeg timed capture failed (exit {rc}); riprovo tra 1s.",
                        flush=True,
                    )
                    if not save_wav and clip.exists():
                        try:
                            clip.unlink()
                        except OSError:
                            pass
                    time.sleep(1.0)
                    continue
                thr = float(args.skip_if_peak_below_dbfs)
                if thr > -119.0:
                    pk = _measure_peak_dbfs(clip)
                    if pk is not None and pk < thr:
                        print(
                            f"[listener] timed record {record_idx} -> skip silenzio "
                            f"(picco {pk:.1f} dBFS < {thr:.0f} dBFS), niente Shazam/TCP",
                            flush=True,
                        )
                        if not save_wav and clip.exists():
                            try:
                                clip.unlink()
                            except OSError:
                                pass
                        record_idx += 1
                        continue
                try:
                    last_policy_sent, last_send_ts = _process_one_clip(
                        clip,
                        matcher=matcher,
                        mapping=mapping,
                        args=args,
                        remove_after_match=remove_after_match,
                        last_policy_sent=last_policy_sent,
                        last_send_ts=last_send_ts,
                    )
                finally:
                    if not save_wav and clip.exists():
                        try:
                            clip.unlink()
                        except OSError:
                            pass
                record_idx += 1
        else:
            seg_dir = watch_dir if save_wav else Path(tempfile.mkdtemp(prefix="robo_shazam_seg_"))
            if not save_wav:
                scrap_segment_dir = seg_dir
            ffmpeg_cmd = _ffmpeg_mic_cmd(
                str(seg_dir / segment_pattern),
                args.segment_seconds,
                args.mic_input,
                sample_rate=int(args.sample_rate),
                no_audio_filter=no_af,
                mic_preset=str(args.mic_preset),
                mic_channels=int(args.mic_channels),
                gain_db=float(args.gain_boost_db),
                auto_normalize=not bool(args.no_auto_normalize),
            )
            ffmpeg = subprocess.Popen(ffmpeg_cmd)
            print("[listener] ffmpeg segment capture started.", flush=True)
            print(
                "[listener] nota: il microfono = suono in STANZA, non il bus audio del Mac; "
                "se vedi salti tra brani con pochi votes, alza --segment-seconds e/o --min-votes.",
                flush=True,
            )

            seen: set[Path] = set()
            while True:
                clips = sorted(
                    p
                    for p in seg_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in (".mp3", ".wav", ".flac", ".m4a")
                )
                for clip in clips:
                    if clip in seen:
                        continue
                    if not _file_stable(clip, args.stability_wait):
                        continue
                    seen.add(clip)
                    thr = float(args.skip_if_peak_below_dbfs)
                    if thr > -119.0:
                        pk = _measure_peak_dbfs(clip)
                        if pk is not None and pk < thr:
                            print(
                                f"[listener] {clip.name} -> skip silenzio "
                                f"(picco {pk:.1f} dBFS < {thr:.0f} dBFS), niente Shazam/TCP",
                                flush=True,
                            )
                            if not save_wav and clip.exists():
                                try:
                                    clip.unlink()
                                except OSError:
                                    pass
                            continue
                    try:
                        last_policy_sent, last_send_ts = _process_one_clip(
                            clip,
                            matcher=matcher,
                            mapping=mapping,
                            args=args,
                            remove_after_match=remove_after_match,
                            last_policy_sent=last_policy_sent,
                            last_send_ts=last_send_ts,
                        )
                    finally:
                        if not save_wav and clip.exists():
                            try:
                                clip.unlink()
                            except OSError:
                                pass

                assert ffmpeg is not None
                if ffmpeg.poll() is not None:
                    raise RuntimeError(f"ffmpeg stopped with code {ffmpeg.returncode}")
                time.sleep(max(0.02, args.poll_interval))
    except KeyboardInterrupt:
        print("\n[listener] stopped by user", flush=True)
    finally:
        if scrap_segment_dir is not None:
            shutil.rmtree(scrap_segment_dir, ignore_errors=True)
        if sd_mic is not None:
            sd_mic.close()
        if ffmpeg is not None and ffmpeg.poll() is None:
            # SIGINT lascia a ffmpeg il tempo di chiudere l'ultimo WAV (header RIFF
            # valido). terminate()/SIGTERM produce invece file da 0 byte tipo
            # mic_000005.wav -> 'Invalid data found when processing input'.
            try:
                ffmpeg.send_signal(signal.SIGINT)
            except OSError:
                ffmpeg.terminate()
            try:
                ffmpeg.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                ffmpeg.kill()


if __name__ == "__main__":
    main()
