from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RhythmFeatures:
    bpm: float
    bpm_std: float
    onset_mean: float
    onset_std: float
    onset_p95: float
    tempogram_peak: float
    tempogram_peak_bpm: float
    tempogram_entropy: float

    def as_vector(self) -> np.ndarray:
        return np.array(
            [
                self.bpm,
                self.bpm_std,
                self.onset_mean,
                self.onset_std,
                self.onset_p95,
                self.tempogram_peak,
                self.tempogram_peak_bpm,
                self.tempogram_entropy,
            ],
            dtype=np.float32,
        )


def extract_rhythm_features(y: np.ndarray, sr: int) -> RhythmFeatures:
    """Extract a compact rhythm/tempo feature vector.

    Designed for short chunks (5-10s). Uses onset strength + tempogram.
    """
    import librosa

    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size == 0 or sr <= 0:
        return RhythmFeatures(0, 0, 0, 0, 0, 0, 0, 0)

    # Normalize peak to reduce loudness sensitivity.
    peak = float(np.max(np.abs(y))) if y.size else 1.0
    if peak > 0:
        y = y / peak

    hop_length = 512
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    oenv = np.asarray(oenv, dtype=np.float32)

    if oenv.size == 0:
        return RhythmFeatures(0, 0, 0, 0, 0, 0, 0, 0)

    onset_mean = float(np.mean(oenv))
    onset_std = float(np.std(oenv))
    onset_p95 = float(np.percentile(oenv, 95))

    # Tempogram (autocorrelation of onset envelope).
    tgram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)
    tgram = np.asarray(tgram, dtype=np.float32)

    # Global tempo estimate (can be multi-valued; keep the strongest).
    tempi = librosa.feature.tempo(onset_envelope=oenv, sr=sr, hop_length=hop_length, aggregate=None)
    tempi = np.asarray(tempi, dtype=np.float32).reshape(-1)
    if tempi.size:
        bpm = float(np.median(tempi))
        bpm_std = float(np.std(tempi))
    else:
        bpm = 0.0
        bpm_std = 0.0

    # Tempogram peak stats
    # librosa tempogram frequencies correspond to BPM bins:
    # https://librosa.org/doc/main/generated/librosa.feature.tempogram.html
    ac_global = np.mean(tgram, axis=1) if tgram.ndim == 2 else tgram.reshape(-1)
    ac_global = np.maximum(ac_global, 0.0)
    # Convert indices to BPM via librosa.tempo_frequencies.
    # Note: the first bin can be inf (0-lag). We ignore non-finite BPM bins.
    bpm_bins = librosa.tempo_frequencies(ac_global.size, sr=sr, hop_length=hop_length)
    bpm_bins = np.asarray(bpm_bins, dtype=np.float32).reshape(-1)
    finite = np.isfinite(bpm_bins)
    if ac_global.size and np.any(finite):
        scores = ac_global.copy()
        scores[~finite] = -1.0
        peak_idx = int(np.argmax(scores))
        peak_val = float(ac_global[peak_idx])
        peak_bpm = float(bpm_bins[peak_idx])
    else:
        peak_idx = 0
        peak_val = 0.0
        peak_bpm = 0.0

    # Entropy of normalized autocorrelation (rhythm complexity)
    p = ac_global / (float(np.sum(ac_global)) + 1e-9)
    ent = float(-np.sum(p * np.log(p + 1e-9)))

    return RhythmFeatures(
        bpm=bpm,
        bpm_std=bpm_std,
        onset_mean=onset_mean,
        onset_std=onset_std,
        onset_p95=onset_p95,
        tempogram_peak=peak_val,
        tempogram_peak_bpm=peak_bpm,
        tempogram_entropy=ent,
    )

