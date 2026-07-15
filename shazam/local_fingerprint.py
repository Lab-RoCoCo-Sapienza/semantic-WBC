"""Hardened Shazam-style local audio fingerprinter.

Design notes:
- Constellation map via `scipy.ndimage.maximum_filter` (Shazam-like peak picking).
- Adaptive amplitude threshold via percentile of the spectrogram magnitude.
- Target zone (time window + frequency window) for forming anchor-target pairs.
- Offset histogram matching with peak-to-mean ratio as confidence.
- Pickle index for compact storage.
"""

import hashlib
import os
import pickle
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.io import wavfile
from scipy.ndimage import maximum_filter
from scipy.signal import stft


@dataclass
class MatchResult:
    song_id: str | None
    song_path: str | None
    confidence: float
    votes: int
    total_hits: int
    offset_sec: float | None
    strategy: str


class LocalShazamMatcher:
    """
    Local landmark-based audio matcher.

    Parameters tuned as a reasonable default for music chunks of 5-10 seconds.
    """

    def __init__(
        self,
        sample_rate: int = 8000,
        nperseg: int = 1024,
        hop_length: int = 256,
        peak_neighborhood: int = 20,
        amp_min_percentile: float = 80.0,
        fanout: int = 10,
        target_zone_time: int = 60,
        target_zone_freq: int = 100,
        min_time_delta_bins: int = 1,
    ) -> None:
        self.sample_rate = sample_rate
        self.nperseg = nperseg
        self.hop_length = hop_length
        self.noverlap = nperseg - hop_length
        self.peak_neighborhood = peak_neighborhood
        self.amp_min_percentile = amp_min_percentile
        self.fanout = fanout
        self.target_zone_time = target_zone_time
        self.target_zone_freq = target_zone_freq
        self.min_time_delta_bins = min_time_delta_bins
        self.hash_db: dict[str, list[tuple[str, int]]] = defaultdict(list)
        self.song_meta: dict[str, dict[str, str]] = {}

    @property
    def seconds_per_frame(self) -> float:
        return self.hop_length / self.sample_rate

    def _to_pcm_wav(self, input_path: str) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-ac", "1",
            "-ar", str(self.sample_rate),
            "-vn",
            "-acodec", "pcm_s16le",
            out_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            if os.path.exists(out_path):
                os.unlink(out_path)
            raise RuntimeError(f"ffmpeg failed for {input_path}:\n{proc.stderr}")
        return out_path

    def _read_mono_audio(self, input_path: str) -> np.ndarray:
        wav_path = self._to_pcm_wav(input_path)
        try:
            sr, audio = wavfile.read(wav_path)
        finally:
            if os.path.exists(wav_path):
                os.unlink(wav_path)
        if sr != self.sample_rate:
            raise ValueError(f"Unexpected sample rate {sr}; expected {self.sample_rate}.")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 0:
            audio = audio / peak
        return audio

    def _spectrogram(self, audio: np.ndarray) -> np.ndarray:
        _, _, zxx = stft(
            audio,
            fs=self.sample_rate,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            boundary=None,
            padded=False,
        )
        return np.log1p(np.abs(zxx))

    def _extract_peaks(self, spec: np.ndarray) -> list[tuple[int, int]]:
        if spec.size == 0:
            return []
        local_max = maximum_filter(spec, size=self.peak_neighborhood) == spec
        threshold = float(np.percentile(spec, self.amp_min_percentile)) if spec.size else 0.0
        detected = local_max & (spec > threshold)
        freq_idx, time_idx = np.where(detected)
        peaks = list(zip(freq_idx.tolist(), time_idx.tolist()))
        peaks.sort(key=lambda p: (p[1], p[0]))
        return peaks

    def _hash_triplet(self, f1: int, f2: int, dt: int) -> str:
        raw = f"{f1}|{f2}|{dt}".encode("ascii")
        return hashlib.sha1(raw).hexdigest()[:20]

    def _fingerprint_peaks(self, peaks: list[tuple[int, int]]) -> list[tuple[str, int]]:
        if len(peaks) < 2:
            return []
        fingerprints: list[tuple[str, int]] = []
        n = len(peaks)
        for i in range(n):
            f1, t1 = peaks[i]
            pairs_made = 0
            j = i + 1
            while j < n and pairs_made < self.fanout:
                f2, t2 = peaks[j]
                dt = t2 - t1
                if dt < self.min_time_delta_bins:
                    j += 1
                    continue
                if dt > self.target_zone_time:
                    break
                if abs(f2 - f1) > self.target_zone_freq:
                    j += 1
                    continue
                fingerprints.append((self._hash_triplet(f1, f2, dt), t1))
                pairs_made += 1
                j += 1
        return fingerprints

    def _fingerprint(self, input_path: str) -> list[tuple[str, int]]:
        audio = self._read_mono_audio(input_path)
        spec = self._spectrogram(audio)
        peaks = self._extract_peaks(spec)
        return self._fingerprint_peaks(peaks)

    def add_song(self, song_path: str) -> str:
        song_id = hashlib.sha1(song_path.encode("utf-8")).hexdigest()[:16]
        title = os.path.splitext(os.path.basename(song_path))[0]
        self.song_meta[song_id] = {"song_path": song_path, "title": title}
        for h, t_anchor in self._fingerprint(song_path):
            self.hash_db[h].append((song_id, t_anchor))
        return song_id

    def build_from_directory(self, songs_dir: str) -> int:
        added = 0
        for name in sorted(os.listdir(songs_dir)):
            if not name.lower().endswith((".mp3", ".wav", ".flac", ".m4a")):
                continue
            song_path = os.path.join(songs_dir, name)
            self.add_song(song_path)
            added += 1
        return added

    def save_index(self, output_path: str) -> None:
        payload = {
            "config": {
                "sample_rate": self.sample_rate,
                "nperseg": self.nperseg,
                "hop_length": self.hop_length,
                "peak_neighborhood": self.peak_neighborhood,
                "amp_min_percentile": self.amp_min_percentile,
                "fanout": self.fanout,
                "target_zone_time": self.target_zone_time,
                "target_zone_freq": self.target_zone_freq,
                "min_time_delta_bins": self.min_time_delta_bins,
            },
            "song_meta": self.song_meta,
            "hash_db": dict(self.hash_db),
        }
        with open(output_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_index(cls, input_path: str) -> "LocalShazamMatcher":
        with open(input_path, "rb") as f:
            payload = pickle.load(f)
        matcher = cls(**payload["config"])
        matcher.song_meta = payload["song_meta"]
        matcher.hash_db = defaultdict(list, payload["hash_db"])
        return matcher

    def match(self, clip_path: str) -> MatchResult:
        query_fps = self._fingerprint(clip_path)
        if not query_fps:
            return MatchResult(None, None, 0.0, 0, 0, None, "no_query_fingerprints")

        per_song_offsets: dict[str, list[int]] = defaultdict(list)
        total_hits = 0
        for h, query_anchor in query_fps:
            candidates = self.hash_db.get(h, [])
            if not candidates:
                continue
            total_hits += len(candidates)
            for song_id, db_anchor in candidates:
                per_song_offsets[song_id].append(db_anchor - query_anchor)

        if not per_song_offsets:
            return MatchResult(None, None, 0.0, 0, total_hits, None, "no_hash_hits")

        best_song_id: str | None = None
        best_votes = 0
        best_offset_bin = 0
        best_ratio = 0.0

        for song_id, offsets in per_song_offsets.items():
            counter = Counter(offsets)
            offset_bin, votes = counter.most_common(1)[0]
            mean = len(offsets) / max(1, len(counter))
            ratio = votes / max(1e-6, mean)
            if votes > best_votes or (votes == best_votes and ratio > best_ratio):
                best_votes = votes
                best_offset_bin = offset_bin
                best_song_id = song_id
                best_ratio = ratio

        if best_song_id is None:
            return MatchResult(None, None, 0.0, 0, total_hits, None, "no_hash_hits")

        offset_sec = round(best_offset_bin * self.seconds_per_frame, 2)
        meta = self.song_meta[best_song_id]
        return MatchResult(
            song_id=best_song_id,
            song_path=meta["song_path"],
            confidence=round(best_ratio, 2),
            votes=best_votes,
            total_hits=total_hits,
            offset_sec=offset_sec,
            strategy=f"hist_peak(votes={best_votes},hits={total_hits})",
        )
