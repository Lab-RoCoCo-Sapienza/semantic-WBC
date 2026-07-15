"""Shazam-driven file-watch controller.

Watches a directory for rolling audio chunks (e.g. 5s mp3 segments), identifies
each new chunk with the local Shazam-style matcher (optionally with a CLAP
similarity fallback), maps the resulting ``song_id``/filename to a policy id via
the user-provided ``song_to_policy`` mapping, and exposes the queued ids to
``RlMultiPolicyPipeline`` through ``pop_pending_policy_id``.

Heavy work (fingerprinting, CLAP) runs in a background thread so the pipeline's
control loop never blocks.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from robojudo.config import ROOT_DIR
from robojudo.controller import Controller, ctrl_registry
from robojudo.controller.ctrl_cfgs import ShazamFileWatchCtrlCfg

logger = logging.getLogger(__name__)


_CHUNK_INDEX_RE = re.compile(r"(\d+)\.(?:mp3|wav|flac|m4a)$", re.IGNORECASE)


@dataclass
class ShazamHit:
    """A single identified chunk, queued for the pipeline to consume.

    Carries enough context to:
    - decide same-song vs different-song at the segment boundary (``song_id``),
    - drop stale/out-of-order queue entries (``chunk_index``, ``ts_mono``),
    - seek the BeyondMimic motion phase to match music (``offset_sec``),
    - delay exposure to the pipeline (``release_at_mono``) to match audio playback.
    """

    policy_id: int
    song_id: str
    song_path: str | None
    offset_sec: float | None
    chunk_index: int | None
    ts_mono: float
    release_at_mono: float = 0.0


def _parse_chunk_index(path: str) -> int | None:
    """Extract trailing numeric index from a chunk filename, e.g. ``chunk_0007.mp3 -> 7``."""
    m = _CHUNK_INDEX_RE.search(os.path.basename(path))
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _ensure_shazam_on_path() -> None:
    """Put ``<repo>/shazam`` on ``sys.path`` so ``local_fingerprint`` is importable."""
    shazam_dir = (ROOT_DIR / "shazam").resolve()
    if shazam_dir.is_dir():
        sp = str(shazam_dir)
        if sp not in sys.path:
            sys.path.insert(0, sp)


@ctrl_registry.register
class ShazamFileWatchCtrl(Controller):
    cfg_ctrl: ShazamFileWatchCtrlCfg

    def __init__(self, cfg_ctrl: ShazamFileWatchCtrlCfg, env=None, device: str = "cpu"):
        super().__init__(cfg_ctrl=cfg_ctrl, env=env, device=device)

        _ensure_shazam_on_path()
        from local_fingerprint import LocalShazamMatcher  # type: ignore  # noqa: E402

        self._matcher_cls = LocalShazamMatcher

        self._watch_dir = self._resolve_path(cfg_ctrl.watch_dir, expect_dir=True, create=True)
        self._index_path = self._resolve_path(cfg_ctrl.index_path, expect_dir=False, create=False)

        self._song_to_policy: dict[str, int] = {
            k.lower(): int(v) for k, v in (cfg_ctrl.song_to_policy or {}).items()
        }

        self._song_time_to_policy: list[tuple[str, float, float | None, int]] = []
        for entry in (getattr(cfg_ctrl, "song_time_to_policy", None) or []):
            try:
                key_raw, start_raw, end_raw, pid_raw = entry
            except (TypeError, ValueError):
                logger.warning(
                    "[ShazamFileWatchCtrl] invalid song_time_to_policy entry %r (expected "
                    "(key, start_s, end_s, policy_id)), skipped.",
                    entry,
                )
                continue
            key = str(key_raw).strip().lower()
            if not key:
                logger.warning(
                    "[ShazamFileWatchCtrl] song_time_to_policy entry has empty key: %r, skipped.",
                    entry,
                )
                continue
            try:
                start_s = float(start_raw)
                end_raw_f = float(end_raw)
                policy_id = int(pid_raw)
            except (TypeError, ValueError):
                logger.warning(
                    "[ShazamFileWatchCtrl] song_time_to_policy entry has non-numeric values: %r, skipped.",
                    entry,
                )
                continue
            if start_s < 0.0:
                logger.warning(
                    "[ShazamFileWatchCtrl] song_time_to_policy entry has negative start: %r, skipped.",
                    entry,
                )
                continue
            end_s: float | None = None if end_raw_f <= 0.0 else end_raw_f
            if end_s is not None and end_s <= start_s:
                logger.warning(
                    "[ShazamFileWatchCtrl] song_time_to_policy range end<=start: %r, skipped.",
                    entry,
                )
                continue
            self._song_time_to_policy.append((key, start_s, end_s, policy_id))
        if self._song_time_to_policy:
            logger.info(
                "[ShazamFileWatchCtrl] timed song->policy rules loaded: %d",
                len(self._song_time_to_policy),
            )

        self._queue: deque[ShazamHit] = deque(maxlen=max(1, int(cfg_ctrl.queue_maxlen)))
        self._lock = threading.Lock()
        self._seen_files: set[str] = set()

        self._stop = threading.Event()

        self._matcher = None
        self._clap_retriever = None
        self._clap_paths: list[str] | None = None
        self._clap_matrix = None
        self._clap_windows = None

        self._watch_name_re: re.Pattern[str] | None = None
        if cfg_ctrl.watch_name_pattern:
            try:
                self._watch_name_re = re.compile(cfg_ctrl.watch_name_pattern)
            except re.error as exc:
                logger.warning(
                    "[ShazamFileWatchCtrl] invalid watch_name_pattern %r: %s",
                    cfg_ctrl.watch_name_pattern,
                    exc,
                )

        self._clean_watch_dir_on_start()

        self._worker = threading.Thread(target=self._run_loop, name="ShazamFileWatch", daemon=True)
        self._worker.start()

        logger.info(
            "[ShazamFileWatchCtrl] watching %s (index=%s, clap_fallback=%s, watch_name_pattern=%s)",
            self._watch_dir,
            self._index_path,
            cfg_ctrl.clap_fallback,
            cfg_ctrl.watch_name_pattern,
        )

    def _clean_watch_dir_on_start(self) -> None:
        """Remove chunk audio left from previous runs so a fresh pipeline does not match stale files."""
        if not bool(getattr(self.cfg_ctrl, "clean_watch_dir_on_start", True)):
            return
        exts = tuple(e.lower() for e in self.cfg_ctrl.supported_exts)
        pat = self._watch_name_re
        removed = 0
        if not self._watch_dir.is_dir():
            return
        for p in sorted(self._watch_dir.iterdir(), key=lambda x: x.name):
            if not p.is_file():
                continue
            name = p.name
            if not name.lower().endswith(exts):
                continue
            if pat is not None and not pat.match(name):
                continue
            try:
                p.unlink()
                removed += 1
            except OSError as e:
                logger.debug("[ShazamFileWatchCtrl] unlink %s: %s", p, e)
        if removed:
            logger.info(
                "[ShazamFileWatchCtrl] cleaned %d chunk file(s) from %s on startup",
                removed,
                self._watch_dir,
            )

    def _resolve_path(self, p: str, *, expect_dir: bool, create: bool) -> Path:
        path = Path(p).expanduser()
        if not path.is_absolute():
            path = (ROOT_DIR / p).resolve()
        if expect_dir:
            if create and not path.exists():
                path.mkdir(parents=True, exist_ok=True)
            if not path.is_dir():
                raise FileNotFoundError(f"ShazamFileWatchCtrl: watch dir not found: {path}")
        return path

    def reset(self):
        with self._lock:
            self._queue.clear()

    def get_data(self):
        with self._lock:
            pending_snapshot = [h.policy_id for h in self._queue]
        return {
            "shazam_pending_policy_ids": pending_snapshot,
        }

    def _head_ready(self, hit: ShazamHit) -> bool:
        """A hit is exposed to the pipeline only after its release time.

        Used to enforce ``controller_delay_s`` so the robot reacts when the user
        actually *hears* the chunk (compensating audio playback latency),
        instead of the moment the streamer drops the file.
        """
        if hit.release_at_mono <= 0.0:
            return True
        return time.monotonic() >= hit.release_at_mono

    def peek_pending(self) -> ShazamHit | None:
        """Return the head of the queue without consuming it.

        Hits whose release time has not arrived yet are hidden (returns None)
        even if the queue is non-empty.
        """
        with self._lock:
            if not self._queue:
                return None
            head = self._queue[0]
            if not self._head_ready(head):
                return None
            return head

    def pop_pending(self) -> ShazamHit | None:
        """Called by the pipeline to consume the oldest queued hit (if released)."""
        with self._lock:
            if not self._queue:
                return None
            head = self._queue[0]
            if not self._head_ready(head):
                return None
            return self._queue.popleft()

    def pop_pending_policy_id(self) -> int | None:
        """Backwards-compatible accessor used by older call sites."""
        hit = self.pop_pending()
        return None if hit is None else hit.policy_id

    def shutdown(self) -> None:
        self._stop.set()

    def __del__(self):
        try:
            self._stop.set()
        except Exception:
            pass

    def _run_loop(self) -> None:
        try:
            self._matcher = self._matcher_cls.load_index(str(self._index_path))
            logger.info("[ShazamFileWatchCtrl] Shazam index loaded: %s", self._index_path)
        except Exception as e:
            logger.error(
                "[ShazamFileWatchCtrl] Failed to load Shazam index %s: %s. "
                "Controller will stay idle (queue empty).",
                self._index_path,
                e,
            )
            return

        if self.cfg_ctrl.clap_fallback and self.cfg_ctrl.clap_index_path:
            self._maybe_load_clap()

        while not self._stop.is_set():
            try:
                self._scan_once()
            except Exception as e:
                logger.exception("[ShazamFileWatchCtrl] Scan error: %s", e)
            self._stop.wait(self.cfg_ctrl.poll_interval_s)

    def _maybe_load_clap(self) -> None:
        clap_json = Path(self.cfg_ctrl.clap_index_path).expanduser()  # type: ignore[arg-type]
        if not clap_json.is_absolute():
            clap_json = (ROOT_DIR / self.cfg_ctrl.clap_index_path).resolve()  # type: ignore[arg-type]
        if not clap_json.is_file():
            logger.warning(
                "[ShazamFileWatchCtrl] CLAP index not found: %s (disabling CLAP fallback).",
                clap_json,
            )
            return
        try:
            from clap_fallback import CLAPRetriever, load_clap_index  # type: ignore
        except Exception as e:
            logger.warning("[ShazamFileWatchCtrl] CLAP deps unavailable (%s); skipping.", e)
            return
        try:
            paths, matrix, meta = load_clap_index(str(clap_json.resolve()))
            model_name = meta.get("model_name", self.cfg_ctrl.clap_model)
            self._clap_retriever = CLAPRetriever(
                model_name=model_name, device=self.cfg_ctrl.clap_device
            )
            self._clap_paths = paths
            self._clap_matrix = matrix
            self._clap_windows = meta.get("windows")
            logger.info("[ShazamFileWatchCtrl] CLAP index loaded: %s (%d items)", clap_json, len(paths))
            try:
                self._clap_retriever.prewarm()
                logger.info("[ShazamFileWatchCtrl] CLAP model weights loaded (prewarmed).")
            except Exception as e:
                logger.warning("[ShazamFileWatchCtrl] CLAP prewarm failed: %s", e)
        except Exception as e:
            logger.warning("[ShazamFileWatchCtrl] CLAP init failed: %s", e)

    def _scan_once(self) -> None:
        if not self._watch_dir.is_dir():
            return
        exts = tuple(e.lower() for e in self.cfg_ctrl.supported_exts)
        names = sorted(os.listdir(self._watch_dir))
        for name in names:
            if not name.lower().endswith(exts):
                continue
            if self._watch_name_re is not None and not self._watch_name_re.match(name):
                continue
            full = str((self._watch_dir / name).resolve())
            if full in self._seen_files:
                continue
            if not self._file_is_stable(full):
                continue
            self._seen_files.add(full)
            self._handle_chunk(full)

    @staticmethod
    def _file_is_stable(path: str) -> bool:
        """Consider a file 'done' if its size is non-zero and unchanged for >=200ms."""
        try:
            s1 = os.path.getsize(path)
        except OSError:
            return False
        if s1 <= 0:
            return False
        time.sleep(0.2)
        try:
            s2 = os.path.getsize(path)
        except OSError:
            return False
        return s1 == s2

    def _handle_chunk(self, path: str) -> None:
        assert self._matcher is not None
        t0 = time.time()
        try:
            result = self._matcher.match(path)
        except Exception as e:
            logger.warning("[ShazamFileWatchCtrl] match(%s) failed: %s", path, e)
            return

        used_clap = False
        if self.cfg_ctrl.clap_fallback and self._clap_retriever is not None:
            try:
                result, used_clap = self._maybe_clap_fallback(result, path)
            except Exception as e:
                logger.warning("[ShazamFileWatchCtrl] CLAP fallback failed: %s", e)

        if result.song_id is None or float(result.confidence) < self.cfg_ctrl.min_confidence:
            logger.info(
                "[ShazamFileWatchCtrl] %s -> no confident match (conf=%.3f, votes=%d, strat=%s, clap=%s)",
                os.path.basename(path),
                float(result.confidence),
                int(result.votes),
                result.strategy,
                used_clap,
            )
            self._maybe_delete(path)
            return

        offset_sec_for_lookup = (
            float(result.offset_sec) if getattr(result, "offset_sec", None) is not None else None
        )
        policy_id, policy_reason = self._resolve_policy_id(
            result.song_id, result.song_path, offset_sec_for_lookup
        )
        dt = (time.time() - t0) * 1000.0
        if policy_id is None:
            logger.info(
                "[ShazamFileWatchCtrl] %s -> song_id=%s song_path=%s (no song_to_policy match, %.1fms)",
                os.path.basename(path),
                result.song_id,
                result.song_path,
                dt,
            )
        else:
            chunk_index = _parse_chunk_index(path)
            offset_sec = offset_sec_for_lookup
            now_mono = time.monotonic()
            delay_s = float(getattr(self.cfg_ctrl, "controller_delay_s", 0.0) or 0.0)
            release_at = now_mono + max(0.0, delay_s)
            hit = ShazamHit(
                policy_id=int(policy_id),
                song_id=str(result.song_id),
                song_path=result.song_path,
                offset_sec=offset_sec,
                chunk_index=chunk_index,
                ts_mono=now_mono,
                release_at_mono=release_at,
            )
            with self._lock:
                evicted = None
                if (
                    self._queue.maxlen is not None
                    and len(self._queue) >= self._queue.maxlen
                ):
                    evicted = self._queue[0]
                self._queue.append(hit)
            if evicted is not None:
                logger.warning(
                    "[ShazamFileWatchCtrl] queue full (maxlen=%d) -> evicted chunk_idx=%s "
                    "song=%s policy=%s",
                    self._queue.maxlen,
                    evicted.chunk_index,
                    evicted.song_id,
                    evicted.policy_id,
                )
            logger.info(
                "[ShazamFileWatchCtrl] %s -> policy %d (song=%s conf=%.3f votes=%d strat=%s clap=%s "
                "offset=%s chunk_idx=%s rule=%s %.1fms)",
                os.path.basename(path),
                policy_id,
                result.song_path,
                float(result.confidence),
                int(result.votes),
                result.strategy,
                used_clap,
                f"{offset_sec:.2f}s" if offset_sec is not None else "n/a",
                chunk_index if chunk_index is not None else "n/a",
                policy_reason,
                dt,
            )

        self._maybe_delete(path)

    def _resolve_policy_id(
        self,
        song_id: str | None,
        song_path: str | None,
        offset_sec: float | None = None,
    ) -> tuple[int | None, str]:
        """Return (policy_id, reason).

        Priority order:
          1. ``song_time_to_policy`` (if ``offset_sec`` is known and range matches)
          2. ``song_to_policy``
        """
        candidates: list[str] = []
        if song_id:
            candidates.append(str(song_id).lower())
        if song_path:
            candidates.append(str(song_path).lower())
            candidates.append(os.path.splitext(os.path.basename(song_path))[0].lower())

        if offset_sec is not None and self._song_time_to_policy:
            for key, start_s, end_s, pid in self._song_time_to_policy:
                if offset_sec < start_s:
                    continue
                if end_s is not None and offset_sec >= end_s:
                    continue
                for c in candidates:
                    if key in c:
                        end_txt = "*" if end_s is None else f"{end_s:g}"
                        return pid, f"timed({key}@{start_s:g}-{end_txt})"

        if self._song_to_policy:
            for key, pid in self._song_to_policy.items():
                for c in candidates:
                    if key in c:
                        return pid, f"plain({key})"

        return None, "no-match"

    def _maybe_clap_fallback(self, r0, clip_path: str):
        """Mirror the logic of ``shazam.run_experiment.maybe_clap_fallback``."""
        try_clap = r0.song_id is None
        ceiling = self.cfg_ctrl.clap_fallback_if_fp_votes_below
        if ceiling is not None and r0.song_id is not None and r0.votes < ceiling:
            try_clap = True
        if not try_clap:
            return r0, False
        if (
            self._clap_retriever is None
            or not self._clap_paths
            or self._clap_matrix is None
            or getattr(self._clap_matrix, "size", 0) == 0
        ):
            return r0, False
        near = self._clap_retriever.nearest(
            clip_path,
            self._clap_paths,
            self._clap_matrix,
            max_seconds=self.cfg_ctrl.clap_embed_seconds,
            windows=self._clap_windows,
        )
        if not near or near.similarity < self.cfg_ctrl.clap_min_sim:
            return r0, False

        from local_fingerprint import MatchResult  # type: ignore

        strategy = (
            f"clap_fallback(cos={near.similarity:.3f}, t={near.offset_sec:.1f}s)"
            if near.offset_sec is not None
            else f"clap_fallback(cos={near.similarity:.3f})"
        )
        return (
            MatchResult(
                song_id=near.song_id,
                song_path=near.song_path,
                confidence=round(float(near.similarity), 4),
                votes=0,
                total_hits=r0.total_hits,
                offset_sec=near.offset_sec,
                strategy=strategy,
            ),
            True,
        )

    def _maybe_delete(self, path: str) -> None:
        if not self.cfg_ctrl.delete_after:
            return
        try:
            os.unlink(path)
        except OSError as e:
            logger.debug("[ShazamFileWatchCtrl] delete failed %s: %s", path, e)
