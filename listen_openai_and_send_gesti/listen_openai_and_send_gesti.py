#!/usr/bin/env python3
"""Microphone listener: OpenAI response -> send GESTI command over TCP."""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import select
import socket
import sys
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

ROOT = Path(__file__).resolve().parents[1]


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--segment-seconds", type=float, default=5.0)
    p.add_argument("--sample-rate", type=int, default=48000)
    p.add_argument("--mic-channels", type=int, choices=(1, 2), default=1)
    p.add_argument("--sd-device", type=int, default=None)
    p.add_argument("--save-dir", default="runtime_chunks_live/gesti_openai")
    p.add_argument("--openai-model", default="gpt-4o-mini-transcribe")
    p.add_argument(
        "--min-transcript-chars",
        type=int,
        default=2,
        help="Attiva invio/speak solo se la trascrizione ha almeno N caratteri non-spazio.",
    )
    p.add_argument("--chat-model", default="gpt-4o-mini")
    p.add_argument(
        "--chat-system-prompt",
        default="Reply briefly and naturally in English.",
        help="System prompt per la risposta AI che verra' poi letta dallo speaker.",
    )
    p.add_argument("--tts-model", default="gpt-4o-mini-tts")
    p.add_argument("--tts-voice", default="alloy")
    p.add_argument(
        "--speak-text-template",
        default="{ai_response}",
        help="Template testo TTS. Placeholder: {text}, {ai_response}.",
    )
    p.add_argument(
        "--no-speak-response",
        action="store_true",
        help="Disabilita riproduzione speaker della risposta TTS.",
    )
    p.add_argument(
        "--speaker-device",
        type=int,
        default=None,
        help="Indice output sounddevice (None = output di default sistema).",
    )

    p.add_argument("--tcp-mode", choices=("client", "server"), default="server")
    p.add_argument("--client-host", default="127.0.0.1")
    p.add_argument("--client-port", type=int, default=8765)
    p.add_argument("--server-host", default="0.0.0.0")
    p.add_argument("--server-port", type=int, default=8765)
    p.add_argument("--connect-timeout", type=float, default=2.0)

    p.add_argument(
        "--out-format",
        default="[GESTI,{tts_seconds:.2f}]",
        help=(
            "Formato riga inviata via TCP. Placeholder disponibili: "
            "{command}, {clip_seconds}, {api_seconds}, {chat_api_seconds}, {tts_seconds}, {text}."
        ),
    )
    p.add_argument("--command-name", default="GESTI")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--debug-mode",
        action="store_true",
        help="Modalita' debug: premendo C invia [GESTI,<debug-seconds>] senza usare mic/OpenAI.",
    )
    p.add_argument(
        "--debug-seconds",
        type=float,
        default=1.30,
        help="Secondi fissi inviati in debug mode (default 1.30 -> [GESTI,1.30]).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


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
    """Line-oriented TCP server that pushes command lines to one client."""

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
            print(f"[gesti-server] client connected: {addr[0]}:{addr[1]}", flush=True)

    def send_line(self, line: str) -> None:
        payload = (line.rstrip("\n") + "\n").encode("utf-8")
        if self.client_sock is None:
            print("[gesti-server] waiting for client connection before send...", flush=True)
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


def record_to_wav(path: Path, sample_rate: int, channels: int, duration_s: float, sd_device: int | None) -> float:
    frames = int(sample_rate * duration_s)
    audio = sd.rec(frames, samplerate=sample_rate, channels=channels, dtype="float32", device=sd_device)
    sd.wait()
    pcm16 = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16.tobytes())

    # Misura robusta dal file scritto.
    with wave.open(str(path), "rb") as w:
        file_frames = w.getnframes()
        rate = w.getframerate()
    return float(file_frames) / float(rate)


def transcribe_with_openai(wav_path: Path, model: str) -> tuple[str, float]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Missing dependency 'openai'. Install with: pip install openai") from exc

    client = OpenAI(api_key=resolve_openai_api_key())
    t0 = time.perf_counter()
    with wav_path.open("rb") as fh:
        result = client.audio.transcriptions.create(
            model=model,
            file=fh,
        )
    elapsed = time.perf_counter() - t0
    text = getattr(result, "text", "") or ""
    return text.strip(), elapsed


def synthesize_tts_wav_with_openai(text: str, model: str, voice: str) -> tuple[bytes, float]:
    from openai import OpenAI

    client = OpenAI(api_key=resolve_openai_api_key())
    t0 = time.perf_counter()
    try:
        # Newer SDK style.
        response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            response_format="wav",
        )
    except TypeError:
        # Backward compatibility for SDKs that don't accept response_format.
        response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
        )
    wav_bytes = response.read()
    elapsed = time.perf_counter() - t0
    return wav_bytes, elapsed


def generate_ai_response_text(user_text: str, model: str, system_prompt: str) -> tuple[str, float]:
    from openai import OpenAI

    client = OpenAI(api_key=resolve_openai_api_key())
    t0 = time.perf_counter()
    result = client.responses.create(
        model=model,
        instructions=system_prompt,
        input=user_text,
    )
    elapsed = time.perf_counter() - t0
    out = (getattr(result, "output_text", "") or "").strip()
    if not out:
        out = "Non ho capito bene, puoi ripetere?"
    return out, elapsed


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


def play_audio_array(audio: np.ndarray, sample_rate: int, speaker_device: int | None) -> float:
    t0 = time.perf_counter()
    sd.play(audio, samplerate=sample_rate, device=speaker_device)
    sd.wait()
    return max(0.0, time.perf_counter() - t0)


def wav_duration_seconds_from_decoded(audio: np.ndarray, sample_rate: int) -> float:
    if sample_rate <= 0:
        return 0.0
    if audio.ndim == 2:
        nframes = int(audio.shape[0])
    else:
        nframes = int(audio.shape[0])
    if nframes <= 0:
        return 0.0
    return float(nframes) / float(sample_rate)


def format_line(
    template: str,
    command_name: str,
    clip_seconds: float,
    api_seconds: float,
    chat_api_seconds: float,
    tts_seconds: float,
    text: str,
) -> str:
    safe_text = text.replace("\n", " ").strip()
    return template.format(
        command=command_name,
        clip_seconds=clip_seconds,
        api_seconds=api_seconds,
        chat_api_seconds=chat_api_seconds,
        tts_seconds=tts_seconds,
        text=safe_text,
    )


def _normalize_text(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _special_routing_from_text(text: str) -> tuple[str | None, str | None]:
    """Ritorna (override_tcp_line, forced_ai_response) se il testo matcha regole speciali."""
    norm = _normalize_text(text)
    if not norm:
        return None, None

    # 1) Introduzione fissa a "what can/you can do"
    if re.search(r"\bwhat\b.*\bcan\b.*\bdo\b", norm) or re.search(
        r"\bwhat\b.*\byou\b.*\bdo\b", norm
    ):
        return None, "i can do Violin, and 67 want to try?"

    # 2) Comandi diretti: se il testo menziona violin o 67 -> manda subito.
    has_violin = bool(
        re.search(r"\bviolin(?:o|i)?\b", norm)
        or re.search(r"\bviolon", norm)
    )
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


def _read_one_key() -> str:
    if os.name == "nt":
        import msvcrt

        ch = msvcrt.getwch()
        return ch or ""

    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        ready, _, _ = select.select([sys.stdin], [], [], 0.1)
        if ready:
            return sys.stdin.read(1)
        return ""
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def run_debug_keyboard_loop(transport: GestureTcpClient | GestureTcpServer, args: argparse.Namespace) -> None:
    debug_line = f"[GESTI,{float(args.debug_seconds):.2f}]"
    print("[gesti-debug] debug mode ON: press 'C' to send, 'Q' to quit.", flush=True)
    while True:
        key = _read_one_key()
        if not key:
            continue
        k = key.lower()
        if k == "q":
            print("[gesti-debug] quit requested.", flush=True)
            return
        if k != "c":
            continue
        if args.dry_run:
            print(f"[gesti-debug] DRY-RUN line={debug_line!r}", flush=True)
            continue
        transport.send_line(debug_line)
        print(f"[gesti-debug] sent line={debug_line!r}", flush=True)


def main() -> None:
    args = parse_args()

    save_dir = Path(args.save_dir)
    if not save_dir.is_absolute():
        save_dir = (ROOT / save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    transport: GestureTcpClient | GestureTcpServer
    if args.tcp_mode == "server":
        transport = GestureTcpServer(
            host=str(args.server_host),
            port=int(args.server_port),
            timeout_s=float(args.connect_timeout),
        )
        transport.start()
        print(
            f"[gesti-server] listening on {args.server_host}:{args.server_port} segment={args.segment_seconds}s",
            flush=True,
        )
    else:
        transport = GestureTcpClient(
            host=str(args.client_host),
            port=int(args.client_port),
            timeout_s=float(args.connect_timeout),
        )
        print(
            f"[gesti-client] target={args.client_host}:{args.client_port} segment={args.segment_seconds}s",
            flush=True,
        )

    print(f"[gesti] saving wav clips in: {save_dir}", flush=True)
    print(f"[gesti] command template: {args.out_format}", flush=True)
    if bool(args.debug_mode):
        try:
            run_debug_keyboard_loop(transport, args)
        except KeyboardInterrupt:
            print("\n[gesti-debug] stopped by user", flush=True)
        finally:
            transport.close()
        return

    try:
        idx = 0
        while True:
            clip = save_dir / f"gesti_{idx:06d}.wav"
            if args.verbose:
                print(
                    f"[gesti] recording clip={clip.name} duration={float(args.segment_seconds):.2f}s...",
                    flush=True,
                )
            clip_seconds = record_to_wav(
                path=clip,
                sample_rate=int(args.sample_rate),
                channels=int(args.mic_channels),
                duration_s=float(args.segment_seconds),
                sd_device=args.sd_device,
            )
            if args.verbose:
                print(f"[gesti] recorded clip={clip.name} clip_s={clip_seconds:.2f}", flush=True)

            try:
                if args.verbose:
                    print(f"[gesti] transcribing clip={clip.name} with OpenAI...", flush=True)
                text, api_seconds = transcribe_with_openai(clip, args.openai_model)
            except Exception as exc:  # noqa: BLE001
                print(f"[gesti] openai error clip={clip.name}: {exc}", flush=True)
                idx += 1
                continue
            if args.verbose:
                preview = text[:80] + ("..." if len(text) > 80 else "")
                print(
                    f"[gesti] openai ok clip={clip.name} api_s={api_seconds:.2f} text={preview!r}",
                    flush=True,
                )

            transcript_len = len((text or "").strip())
            if transcript_len < int(args.min_transcript_chars):
                if args.verbose:
                    print(
                        f"[gesti] skip: transcript too short ({transcript_len} < {int(args.min_transcript_chars)}).",
                        flush=True,
                    )
                idx += 1
                continue

            chat_api_seconds = 0.0
            ai_response_text = ""
            override_line, forced_reply = _special_routing_from_text(text)

            if forced_reply is not None:
                ai_response_text = forced_reply
                if args.verbose:
                    print(
                        f"[gesti] special rule matched -> reply={ai_response_text!r} "
                        f"line_override={override_line!r}",
                        flush=True,
                    )
            try:
                if forced_reply is None:
                    if args.verbose:
                        print(f"[gesti] generating AI reply from text={text!r}", flush=True)
                    ai_response_text, chat_api_seconds = generate_ai_response_text(
                        user_text=text or "silenzio",
                        model=str(args.chat_model),
                        system_prompt=str(args.chat_system_prompt),
                    )
                    if args.verbose:
                        print(
                            f"[gesti] ai reply ok chat_api_s={chat_api_seconds:.2f} reply={ai_response_text!r}",
                            flush=True,
                        )
            except Exception as exc:  # noqa: BLE001
                ai_response_text = "Scusa, c'e' stato un errore."
                print(f"[gesti] ai-reply error clip={clip.name}: {exc}", flush=True)

            tts_seconds = 0.0
            tts_api_seconds = 0.0
            tts_wav: bytes | None = None
            if not bool(args.no_speak_response):
                spoken_text = str(args.speak_text_template).format(
                    text=text or "silenzio",
                    ai_response=ai_response_text,
                )
                try:
                    if args.verbose:
                        print(f"[gesti] synthesizing TTS text={spoken_text!r}", flush=True)
                    tts_wav, tts_api_seconds = synthesize_tts_wav_with_openai(
                        spoken_text,
                        model=str(args.tts_model),
                        voice=str(args.tts_voice),
                    )
                    audio_tts, sr_tts = decode_wav_bytes(tts_wav)
                    tts_seconds = wav_duration_seconds_from_decoded(audio_tts, sr_tts)
                    if tts_seconds <= 0.0 or tts_seconds > 120.0:
                        # Fallback conservativo: stima da testo (circa 2.7 parole/s).
                        words = len((spoken_text or "").split())
                        tts_seconds = max(0.8, words / 2.7)
                except Exception as exc:  # noqa: BLE001
                    tts_wav = None
                    tts_seconds = 0.0
                    tts_api_seconds = 0.0
                    print(f"[gesti] tts/speaker error clip={clip.name}: {exc}", flush=True)

            line = format_line(
                template=str(args.out_format),
                command_name=str(args.command_name),
                clip_seconds=clip_seconds,
                api_seconds=api_seconds,
                chat_api_seconds=chat_api_seconds,
                tts_seconds=tts_seconds,
                text=text,
            )
            if override_line is not None:
                line = override_line

            if args.dry_run:
                print(
                    f"[gesti] DRY-RUN line={line!r} clip_s={clip_seconds:.2f} api_s={api_seconds:.2f}",
                    flush=True,
                )
            else:
                transport.send_line(line)
                if args.verbose:
                    print(
                        f"[gesti] sent line={line!r} clip={clip.name} clip_s={clip_seconds:.2f} "
                        f"api_s={api_seconds:.2f} text={text!r}",
                        flush=True,
                    )
                else:
                    print(
                        f"[gesti] sent line={line!r} clip_s={clip_seconds:.2f} api_s={api_seconds:.2f}",
                        flush=True,
                    )

            if tts_wav is not None and not bool(args.no_speak_response):
                try:
                    audio_tts, sr_tts = decode_wav_bytes(tts_wav)
                    playback_s = play_audio_array(audio_tts, sr_tts, speaker_device=args.speaker_device)
                    if args.verbose:
                        print(
                            f"[gesti] speaker playback ok tts_s={playback_s:.2f} "
                            f"tts_audio_s={tts_seconds:.2f} tts_api_s={tts_api_seconds:.2f}",
                            flush=True,
                        )
                except Exception as exc:  # noqa: BLE001
                    print(f"[gesti] tts/speaker error clip={clip.name}: {exc}", flush=True)
            idx += 1
    except KeyboardInterrupt:
        print("\n[gesti] stopped by user", flush=True)
    finally:
        transport.close()


if __name__ == "__main__":
    main()
