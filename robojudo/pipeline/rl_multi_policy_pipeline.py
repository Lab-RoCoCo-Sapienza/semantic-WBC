import logging
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import robojudo.environment
import robojudo.policy
from robojudo.controller import CtrlManager
from robojudo.environment import Environment
from robojudo.environment.mujoco_viewer_progress import (
    install_mujoco_viewer_progress_hook,
    set_mujoco_viewer_progress_state,
)
from robojudo.pipeline import Pipeline, pipeline_registry
from robojudo.pipeline.pipeline_cfgs import RlMultiPolicyPipelineCfg
from robojudo.pipeline.rl_pipeline import PolicyWrapper, RlPipeline
from robojudo.policy import PolicyCfg
from robojudo.utils.step_timer import StepTimer

logger = logging.getLogger(__name__)


def _wall_clock_ts() -> str:
    now = datetime.now()
    return now.strftime("%m-%d %H:%M:%S.") + f"{now.microsecond // 1000:03d}"


class PolicyManager:
    DELAY_STEPS_SWITCH: int = 10  # steps to wait before switching policy
    DELAY_STEPS_SWITCH_FAST: int = 1  # steps to wait for low-latency direct switches (mimic->mimic)

    def __init__(
        self,
        cfg_policies: list[PolicyCfg],
        env: Environment,
        device: str = "cpu",
    ):
        self.env = env
        self.device = device

        self.policies: list[PolicyWrapper] = []
        for cfg_policy in cfg_policies:
            policy_entry = PolicyWrapper(cfg_policy, self.env.dof_cfg, device)
            self.policies.append(policy_entry)

        self._current_policy_id: int = 0
        self.warmup_policy_indices = set()

        self.timer = StepTimer()

    @property
    def current_policy_id(self):
        return self._current_policy_id

    @property
    def num_policies(self):
        return len(self.policies)

    @property
    def policy(self) -> PolicyWrapper:
        return self.policies[self.current_policy_id]

    def policy_by_id(self, policy_id) -> PolicyWrapper:
        if not (0 <= policy_id < self.num_policies):
            raise ValueError(f"Policy id {policy_id} out of range [0, {self.num_policies})")
        return self.policies[policy_id]

    def set_policy(self, policy_id: int):
        """Instantly set the policy as policy_id."""
        if not (0 <= policy_id < self.num_policies):
            raise ValueError(f"Policy id {policy_id} out of range [0, {self.num_policies})")
        self.warmup_policy_indices.discard(policy_id)

        self._current_policy_id = policy_id
        # refresh env
        self.env.reset()
        self.env.update_dof_cfg(override_cfg=self.policy.cfg_action_dof)
        logger.warning(f"Switched to policy: {policy_id}: {self.policy.name}")

    def switch_policy(self, policy_id: int, *, fast: bool = False):
        """Switch to the policy as policy_id after delay.

        ``fast=True`` uses ``DELAY_STEPS_SWITCH_FAST`` (≈one env step) instead of the
        default 10 steps; intended for direct mimic→mimic transitions where we don't
        need to drain a long queue of pending callbacks before swapping in the new ONNX.
        """
        if not (0 <= policy_id < self.num_policies):
            raise ValueError(f"Policy id {policy_id} out of range [0, {self.num_policies})")
        self.policy_by_id(policy_id).reset()
        self.warmup_policy_indices.add(policy_id)
        delay = self.DELAY_STEPS_SWITCH_FAST if fast else self.DELAY_STEPS_SWITCH
        self.timer.add(lambda: self.set_policy(policy_id), delay_steps=delay)

    def step(self, env_data, ctrl_data):
        # policy warmup
        for idx in self.warmup_policy_indices:
            if idx != self.current_policy_id:
                self.policy_by_id(idx).get_observation(env_data, ctrl_data)

        self.timer.tick()


@pipeline_registry.register
class RlMultiPolicyPipeline(RlPipeline):
    cfg: RlMultiPolicyPipelineCfg

    @property
    def policy(self) -> PolicyWrapper:
        return self.policy_manager.policy

    def __init__(self, cfg: RlMultiPolicyPipelineCfg):
        # Skip RlPipeline initialization
        Pipeline.__init__(self, cfg=cfg)

        env_class: type[Environment] = getattr(robojudo.environment, self.cfg.env.env_type)
        self.env: Environment = env_class(cfg_env=self.cfg.env, device=self.device)

        self.ctrl_manager = CtrlManager(cfg_ctrls=self.cfg.ctrl, env=self.env, device=self.device)
        install_mujoco_viewer_progress_hook(getattr(self.env, "viewer", None))

        self.policy_manager = PolicyManager(
            cfg_policies=self.cfg.policies,
            env=self.env,
            device=self.device,
        )
        self.env.update_dof_cfg(override_cfg=self.policy.cfg_action_dof)
        self.visualizer = self.env.visualizer

        self.freq = self.cfg.policies[0].freq
        self.dt = 1.0 / self.freq
        self._beyondmimic_stand_hold_s: float = self.cfg.beyondmimic_stand_hold_s

        # State used by `reset()` / priming logic must exist before `self_check()`.
        self._motion_done_switch_queued: bool = False
        self._stand_policy_id: int | None = self._detect_stand_policy_id()
        self._pending_beyondmimic_id: int | None = None
        self._beyondmimic_prime_steps_remaining: int = 0

        self._segment_pid: int | None = None
        self._segment_t0_mono: float = 0.0
        self._segment_t0_wall: str = ""
        self._segment_last_log_mono: float = 0.0
        self._segment_switch_fired: bool = False
        self._pending_safe_window_policy_id: int | None = None

        self._segment_duration_by_pid: dict[int, float] = {
            i: float(getattr(p, "segment_ratio", 1.0)) * float(getattr(self.cfg, "segment_beat_s", 5.0))
            for i, p in enumerate(self.cfg.policies)
        }

        # Shazam synchronization state (Phase 1-5).
        # ``_current_song_id`` is the song backing the currently-running BeyondMimic
        # policy (or None when on Walk/Stand). ``_last_applied_hit`` is the most
        # recent ShazamHit we consumed (used to drop stale/out-of-order entries).
        # ``_boundary_consumed_this_step`` prevents multiple sources (Shazam,
        # MOTION_DONE, time-box) from emitting switches in the same step.
        self._current_song_id: str | None = None
        self._last_applied_hit = None  # type: object | None
        self._boundary_consumed_this_step: bool = False

        self._remote_motion_deadline_mono: float | None = None
        self._remote_motion_return_policy_id: int = int(
            getattr(self.cfg, "motion_done_fallback_policy_id", 0)
        )

        self.self_check()
        self.reset()

    def _detect_stand_policy_id(self) -> int | None:
        explicit = getattr(self.cfg, "beyondmimic_prime_policy_id", None)
        if explicit is not None:
            explicit = int(explicit)
            if 0 <= explicit < self.policy_manager.num_policies:
                return explicit
            logger.warning("Ignoring invalid beyondmimic_prime_policy_id=%s", explicit)
        for idx, p in enumerate(self.cfg.policies):
            if getattr(p, "policy_type", None) == "UnitreeWoGaitPolicy":
                return idx
        return None

    def _save_safe_snapshot(self) -> None:
        """Persist a lightweight runtime snapshot for safe recovery."""
        path = str(getattr(self.cfg, "safe_snapshot_path", "") or "").strip()
        if not path:
            logger.warning("SAFE snapshot skipped: empty safe_snapshot_path.")
            return

        cur_pid = self.policy_manager.current_policy_id
        policy_name = self.policy_manager.policy_by_id(cur_pid).name
        snapshot: dict[str, object] = {
            "saved_at_wall": _wall_clock_ts(),
            "pipeline_timestep": int(self.timestep),
            "current_policy_id": int(cur_pid),
            "current_policy_name": str(policy_name),
        }

        inner = getattr(self.policy, "policy", None)
        if inner is not None and hasattr(inner, "timestep"):
            try:
                snapshot["policy_timestep"] = float(getattr(inner, "timestep"))
            except Exception:
                pass
        if inner is not None and hasattr(inner, "play_speed"):
            try:
                snapshot["policy_play_speed"] = float(getattr(inner, "play_speed"))
            except Exception:
                pass

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2)
            f.write("\n")
        logger.warning("SAFE snapshot saved -> %s", path)

        if bool(getattr(self.cfg, "safe_snapshot_keep_history", True)):
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            history_path = target.parent / f"safe_step_{stamp}.json"
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2)
                f.write("\n")
            logger.warning("SAFE snapshot history saved -> %s", history_path.as_posix())

    def _reset_policy_runtime_overrides(self, policy_id: int) -> None:
        """Restore runtime-overridden policy limits to cfg defaults before a normal switch."""
        wrapper = self.policy_manager.policy_by_id(policy_id)
        inner = getattr(wrapper, "policy", None)
        cfg = getattr(inner, "cfg_policy", None) if inner is not None else None
        if inner is None or cfg is None:
            return
        if hasattr(inner, "max_timestep") and hasattr(cfg, "max_timestep"):
            inner.max_timestep = int(getattr(cfg, "max_timestep"))
        if hasattr(inner, "play_speed_default") and hasattr(inner, "play_speed"):
            inner.play_speed = float(getattr(inner, "play_speed_default"))
        if hasattr(inner, "flag_motion_done"):
            inner.flag_motion_done = False

    def _apply_safe_window(self, policy_id: int) -> bool:
        """Apply configured SAFE step window to a policy (if available)."""
        window = self.cfg.safe_policy_windows.get(policy_id)
        if window is None:
            logger.warning("SAFE window missing for policy_id=%d", policy_id)
            return False
        start_step, end_step = int(window[0]), int(window[1])
        if end_step <= start_step:
            logger.warning(
                "SAFE window invalid for policy_id=%d (start=%d, end=%d)",
                policy_id,
                start_step,
                end_step,
            )
            return False

        wrapper = self.policy_manager.policy_by_id(policy_id)
        inner = getattr(wrapper, "policy", None)
        if inner is None or not hasattr(inner, "timestep") or not hasattr(inner, "max_timestep"):
            logger.warning("SAFE window ignored: policy_id=%d does not support timestep/max_timestep", policy_id)
            return False

        if hasattr(inner, "reset"):
            inner.reset()
        inner.timestep = float(start_step)
        inner.max_timestep = int(end_step)
        if hasattr(inner, "play_speed_default") and hasattr(inner, "play_speed"):
            inner.play_speed = float(getattr(inner, "play_speed_default"))
        if hasattr(inner, "flag_motion_done"):
            inner.flag_motion_done = False

        logger.warning(
            "SAFE window applied policy_id=%d (%s): start=%d end=%d",
            policy_id,
            wrapper.name,
            start_step,
            end_step,
        )
        return True

    def _is_beyondmimic_policy_id(self, policy_id: int) -> bool:
        if not (0 <= policy_id < len(self.cfg.policies)):
            return False
        return getattr(self.cfg.policies[policy_id], "policy_type", None) == "BeyondMimicPolicy"

    def _expand_listener_commands(self, commands: list[str]) -> list[str]:
        """Translate smart-mic listener tokens into ``[POLICY_SWITCH]`` commands."""
        token_map: dict[str, int] = dict(getattr(self.cfg, "listener_token_policy_map", None) or {})
        gesti_pid = getattr(self.cfg, "listener_gesti_policy_id", None)
        fallback = int(getattr(self.cfg, "motion_done_fallback_policy_id", 0))
        expanded: list[str] = []

        for cmd in commands:
            stripped = str(cmd).strip()
            if not stripped:
                continue

            gesti_match = re.match(r"^\[GESTI\s*,\s*([0-9.]+)\s*\]$", stripped, flags=re.IGNORECASE)
            if gesti_match is not None and gesti_pid is not None:
                duration_s = max(0.1, min(float(gesti_match.group(1)), 120.0))
                self._remote_motion_deadline_mono = time.monotonic() + duration_s
                self._remote_motion_return_policy_id = fallback
                expanded.append(f"[POLICY_SWITCH],{int(gesti_pid)}")
                logger.info(
                    "Listener GESTI -> policy %d for %.2fs then %d",
                    int(gesti_pid),
                    duration_s,
                    fallback,
                )
                continue

            if stripped.startswith("[") and stripped.endswith("]"):
                token = stripped[1:-1].strip()
                mapped_pid: int | None = None
                for key, pid in token_map.items():
                    if token.lower() == str(key).lower():
                        mapped_pid = int(pid)
                        break
                if mapped_pid is not None:
                    expanded.append(f"[POLICY_SWITCH],{mapped_pid}")
                    logger.info("Listener token %s -> policy %d", stripped, mapped_pid)
                    continue

            expanded.append(cmd)

        return expanded

    def _maybe_apply_remote_motion_timer(self, commands: list[str]) -> None:
        if self._remote_motion_deadline_mono is None:
            return
        if time.monotonic() < self._remote_motion_deadline_mono:
            return
        self._remote_motion_deadline_mono = None
        ret_pid = int(self._remote_motion_return_policy_id)
        if not (0 <= ret_pid < self.policy_manager.num_policies):
            ret_pid = 0
        if self.policy_manager.current_policy_id != ret_pid:
            commands.append(f"[POLICY_SWITCH],{ret_pid}")
            logger.info("Remote motion timer elapsed -> policy %d", ret_pid)

    def _clear_beyondmimic_prime(self):
        self._pending_beyondmimic_id = None
        self._beyondmimic_prime_steps_remaining = 0

    def _beyondmimic_stand_hold_steps(self) -> int:
        return max(1, int(round(self._beyondmimic_stand_hold_s * self.freq)))

    def _request_policy_switch(self, policy_id: int, *, skip_stand_prime: bool = False):
        """Apply policy switch with optional Stand priming before BeyondMimic.

        When ``skip_stand_prime`` is True, go directly to ``policy_id`` without
        routing through Stand or the ``beyondmimic_stand_hold_s`` hold. Used for
        explicit/direct commands that intentionally bypass the normal Stand
        priming path.

        If config ``beyondmimic_skip_stand_prime`` is True, the same direct path is
        used even for ``[POLICY_SWITCH]`` (e.g. keyboard from Walk to a dance).
        """
        if not (0 <= policy_id < self.policy_manager.num_policies):
            return

        skip_prime = skip_stand_prime or self.cfg.beyondmimic_skip_stand_prime

        if skip_prime:
            # Cancel any staged mimic so the priming state machine doesn't fight us.
            self._clear_beyondmimic_prime()
            if self.policy_manager.current_policy_id != policy_id:
                self._reset_policy_runtime_overrides(policy_id)
                logger.info(
                    f"Direct policy switch (no Stand prime, fast) -> {policy_id}: "
                    f"{self.policy_manager.policy_by_id(policy_id).name}"
                )
                self.policy_manager.switch_policy(policy_id, fast=True)
            return

        stand_id = self._stand_policy_id
        # Cancel any staged mimic if the user explicitly selects another policy,
        # including Stand itself. Without this, a pending prime can survive a
        # later "[POLICY_SWITCH],1" and re-launch the queued dance shortly after
        # we visually land on Stand.
        if self._pending_beyondmimic_id is not None and policy_id != self._pending_beyondmimic_id:
            self._clear_beyondmimic_prime()

        if self._is_beyondmimic_policy_id(policy_id) and stand_id is not None:
            if self.policy_manager.current_policy_id != stand_id:
                self._pending_beyondmimic_id = policy_id
                self._beyondmimic_prime_steps_remaining = 0
                logger.info(f"Priming Stand (policy {stand_id}) before BeyondMimic (policy {policy_id}).")
                # fast=True: do not also queue DELAY_STEPS_SWITCH(10) here; each sim step is costly in
                # wall time when the viewer is open, and that delay stacks on top of stand_hold_s.
                self.policy_manager.switch_policy(stand_id, fast=True)
                return
            # Already on stand: same wall-clock hold, then BeyondMimic.
            self._pending_beyondmimic_id = policy_id
            self._beyondmimic_prime_steps_remaining = 0
            logger.info(
                f"Holding Stand {self._beyondmimic_stand_hold_s:g}s before BeyondMimic (policy {policy_id})."
            )
            return

        self._reset_policy_runtime_overrides(policy_id)
        self.policy_manager.switch_policy(policy_id)

    def _shazam_hud_lines(self) -> list[tuple[str, str]] | None:
        """Two-row bottom-right HUD: last matched chunk vs current motion segment."""
        if self._get_shazam_ctrl() is None:
            return None
        cur_pid = self.policy_manager.current_policy_id
        pname = self.policy_manager.policy_by_id(cur_pid).name
        inner = getattr(self.policy, "policy", None)
        play_speed = float(getattr(inner, "play_speed", 1.0) or 1.0)
        dur = float(self._segment_duration_by_pid.get(cur_pid, 0.0))
        remain = None
        if self._segment_pid == cur_pid and dur > 0.0:
            elapsed = time.monotonic() - self._segment_t0_mono
            remain = max(0.0, dur - elapsed)

        last = self._last_applied_hit
        peek = self._peek_shazam_hit()

        if last is None:
            pred_r = "nessun chunk ancora"
            if peek is not None:
                pci = getattr(peek, "chunk_index", None)
                pred_r += (
                    f" | in coda #{int(pci)}" if pci is not None else " | in coda (idx n/a)"
                )
        else:
            ci = getattr(last, "chunk_index", None)
            spath = getattr(last, "song_path", None)
            sid = getattr(last, "song_id", None)
            chunk_part = f"{int(ci)}" if ci is not None else "?"
            if spath:
                hint = os.path.splitext(os.path.basename(str(spath)))[0][:40]
            elif sid:
                hint = str(sid)[:40]
            else:
                hint = "?"
            pred_r = f"chunk #{chunk_part} -> {hint}"
            if peek is not None:
                pci = getattr(peek, "chunk_index", None)
                lci = getattr(last, "chunk_index", None)
                if pci is not None and (lci is None or int(pci) != int(lci)):
                    pred_r += f" | in coda #{int(pci)}"

        if remain is not None and dur > 0.0:
            mot_r = f"motion {pname} | speed x{play_speed:.2f} | ancora {remain:.1f}s su {dur:.1f}s"
        else:
            mot_r = f"motion {pname} | speed x{play_speed:.2f} | durata non a tempo (switch manuale)"

        return [
            ("Chunk predetto ->", pred_r),
            ("Sto eseguendo ->", mot_r),
        ]

    def _update_mujoco_viewer_progress(self):
        inner = getattr(self.policy, "policy", None)
        progress = None
        if inner is not None and hasattr(inner, "motion_progress_for_gui"):
            try:
                progress = inner.motion_progress_for_gui()
            except Exception:
                progress = None

        payload = {
            "title": "Policy progress",
            "phase": f"id={self.policy_manager.current_policy_id}",
            "policy": getattr(self.policy, "name", "policy"),
            "detail": "-",
            "fraction": None,
            "eta_s": None,
        }
        if isinstance(progress, dict):
            payload["policy"] = progress.get("policy", payload["policy"])
            payload["detail"] = progress.get("detail", payload["detail"])
            payload["fraction"] = progress.get("fraction", payload["fraction"])
            payload["eta_s"] = progress.get("eta_s", payload["eta_s"])
            mode = progress.get("mode", "")
            if mode:
                payload["phase"] = f"id={self.policy_manager.current_policy_id} | {mode}"
        else:
            payload["detail"] = "no time limit for this policy (switch manually)"

        legend = getattr(self.cfg, "mujoco_overlay_legend", None)
        if legend:
            payload["legend_rows"] = legend

        hud_lines = self._shazam_hud_lines()
        if hud_lines:
            payload["shazam_hud"] = {"show": True, "lines": hud_lines}

        set_mujoco_viewer_progress_state(getattr(self.env, "viewer", None), payload)

    def _log_policy_segment_wallclock(self, pid: int, skip_ids: list[int]) -> None:
        """Log policy wall-clock segment start/end and periodic elapsed time (not inference ms)."""
        log_iv = self.cfg.policy_inference_log_interval_s
        now_m = time.monotonic()
        now_w = _wall_clock_ts()

        def _end_segment(reason: str) -> None:
            if self._segment_pid is None:
                return
            old_id = self._segment_pid
            name = self.policy_manager.policy_by_id(old_id).name
            elapsed = now_m - self._segment_t0_mono
            logger.info(
                "Policy segment END (%s) id=%d %s start_wall=%s end_wall=%s duration_s=%.3f",
                reason,
                old_id,
                name,
                self._segment_t0_wall,
                now_w,
                elapsed,
            )
            self._segment_pid = None

        if pid in skip_ids:
            _end_segment("skipped_loco_policy")
            return

        if self._segment_pid != pid:
            if self._segment_pid is not None:
                _end_segment("policy_switch")
            self._segment_pid = pid
            self._segment_t0_mono = now_m
            self._segment_t0_wall = now_w
            self._segment_last_log_mono = now_m
            self._segment_switch_fired = False
            logger.info(
                "Policy segment START wall=%s id=%d %s (excluded_from_logs=%s)",
                now_w,
                pid,
                self.policy.name,
                skip_ids,
            )
            return

        if log_iv > 0 and now_m - self._segment_last_log_mono >= log_iv:
            self._segment_last_log_mono = now_m
            elapsed = now_m - self._segment_t0_mono
            logger.info(
                "Policy segment elapsed id=%d %s started_wall=%s now_wall=%s elapsed_s=%.3f",
                pid,
                self.policy.name,
                self._segment_t0_wall,
                now_w,
                elapsed,
            )

    def self_check(self):
        self.policy_manager.warmup_policy_indices = set(list(range(self.policy_manager.num_policies)))
        super().self_check()
        self.policy_manager.warmup_policy_indices = set()

    def reset(self):
        initial_pid = int(getattr(self.cfg, "initial_policy_id", 0))
        if 0 <= initial_pid < self.policy_manager.num_policies:
            # Make the desired initial policy active before the generic reset path
            # so startup does not briefly run/visualize Walk and only afterwards
            # jump to Stand.
            self.policy_manager._current_policy_id = initial_pid
            self.env.update_dof_cfg(override_cfg=self.policy.cfg_action_dof)

        super().reset()
        self.policy_manager.timer.clear()
        self._clear_beyondmimic_prime()
        self._motion_done_switch_queued = False
        self._segment_pid = None
        self._segment_switch_fired = False
        self._current_song_id = None
        self._last_applied_hit = None
        self._boundary_consumed_this_step = False
        logger.info(
            "Initial policy active after reset: %d: %s",
            self.policy_manager.current_policy_id,
            self.policy.name,
        )

    # ------------------------------------------------------------------
    # Shazam helpers
    # ------------------------------------------------------------------
    def _get_shazam_ctrl(self):
        """Return the ShazamFileWatchCtrl instance (or None if not configured)."""
        try:
            box = self.ctrl_manager.controllers
            if "ShazamFileWatchCtrl" not in box:
                return None
            return box["ShazamFileWatchCtrl"].inst
        except Exception:
            return None

    def _peek_shazam_hit(self):
        """Non-destructive look at the head of the Shazam queue."""
        ctrl = self._get_shazam_ctrl()
        if ctrl is None or not hasattr(ctrl, "peek_pending"):
            return None
        return ctrl.peek_pending()

    def _pop_shazam_hit(self):
        """Consume the head of the Shazam queue, guarding policy_id bounds."""
        ctrl = self._get_shazam_ctrl()
        if ctrl is None or not hasattr(ctrl, "pop_pending"):
            return None
        hit = ctrl.pop_pending()
        if hit is None:
            return None
        pid = getattr(hit, "policy_id", None)
        if pid is None or not (0 <= int(pid) < self.policy_manager.num_policies):
            return None
        return hit

    def _is_current_song_stale(self) -> bool:
        """Return True when the most recent Shazam hit is older than ``max_chunk_age_s``.

        Used to decide whether the music is "still playing" so we can loop a
        BeyondMimic motion in place rather than falling back to Walk.
        """
        last = self._last_applied_hit
        if last is None:
            return True
        ctrl = self._get_shazam_ctrl()
        cfg = getattr(ctrl, "cfg_ctrl", None) if ctrl is not None else None
        max_age = float(getattr(cfg, "max_chunk_age_s", 0.0) or 0.0)
        if max_age <= 0.0:
            # Without a configured TTL, give the song a generous 8s grace window.
            max_age = 8.0
        ts = float(getattr(last, "ts_mono", 0.0))
        if ts <= 0.0:
            return False
        return (time.monotonic() - ts) > max_age

    def _drop_stale_hits(self) -> None:
        """Drop queue heads that are too old or out-of-order wrt the last applied hit."""
        ctrl = self._get_shazam_ctrl()
        if ctrl is None or not hasattr(ctrl, "peek_pending"):
            return
        cfg = getattr(ctrl, "cfg_ctrl", None)
        max_age = float(getattr(cfg, "max_chunk_age_s", 0.0) or 0.0)
        drop_stale = bool(getattr(cfg, "drop_stale_chunks", True))
        last_idx: int | None = None
        if self._last_applied_hit is not None:
            ci = getattr(self._last_applied_hit, "chunk_index", None)
            if ci is not None:
                last_idx = int(ci)
        now = time.monotonic()
        while True:
            hit = ctrl.peek_pending()
            if hit is None:
                return
            ts = float(getattr(hit, "ts_mono", now))
            ci = getattr(hit, "chunk_index", None)
            too_old = max_age > 0.0 and (now - ts) > max_age
            out_of_order = (
                drop_stale
                and last_idx is not None
                and ci is not None
                and int(ci) <= last_idx
            )
            if not (too_old or out_of_order):
                return
            dropped = ctrl.pop_pending()
            if dropped is None:
                return
            logger.info(
                "Shazam: dropped stale hit song=%s chunk_idx=%s age=%.1fs (too_old=%s, out_of_order=%s).",
                getattr(dropped, "song_id", None),
                getattr(dropped, "chunk_index", None),
                now - float(getattr(dropped, "ts_mono", now)),
                too_old,
                out_of_order,
            )

    def _apply_phase_seek(self, hit) -> None:
        """Seek the running BeyondMimic motion to match the music phase.

        Motion nominal duration is derived from ``segment_ratio * segment_beat_s``
        for the current policy. We map ``offset_sec`` (position in the source
        song) to a fractional phase and write it back into the policy's internal
        ``timestep``, resetting the auto-end detector so MOTION_DONE does not
        fire spuriously right after the seek.
        """
        inner = getattr(self.policy, "policy", None)
        offset_sec = getattr(hit, "offset_sec", None)
        if inner is None or offset_sec is None or not hasattr(inner, "timestep"):
            return
        pid = self.policy_manager.current_policy_id
        dur = self._segment_duration_by_pid.get(pid, 0.0)
        if dur <= 0.0:
            return
        # Wrap the music offset into the motion duration so short motions
        # repeat and long motions clamp to their natural length.
        frac = (float(offset_sec) % dur) / dur
        freq = float(getattr(self, "freq", 50.0) or 50.0)
        # `inner.timestep` advances by `play_speed` each env step, so the total amount
        # the timestep accumulates over `dur` real seconds is `dur*freq*play_speed`.
        # Phase seek targets the same coordinate system.
        play_speed = float(getattr(inner, "play_speed", 1.0) or 1.0)
        total_steps = dur * freq * play_speed
        start_ts = 0.0
        cfg_policy = getattr(inner, "cfg_policy", None)
        if cfg_policy is not None:
            start_ts = float(getattr(cfg_policy, "start_timestep", 0.0) or 0.0)
        target = start_ts + frac * total_steps
        inner.timestep = target
        if hasattr(inner, "_auto_estimated_end_timestep"):
            inner._auto_estimated_end_timestep = None
        if hasattr(inner, "_auto_stable_count"):
            inner._auto_stable_count = 0
        if hasattr(inner, "_auto_seen_motion"):
            # Avoid an immediate MOTION_DONE from a post-seek stable burst.
            inner._auto_seen_motion = True
        logger.info(
            "Phase seek pid=%d song=%s offset=%.2fs -> timestep=%.1f/%.1f (frac=%.2f)",
            pid,
            getattr(hit, "song_id", None),
            float(offset_sec),
            target,
            total_steps,
            frac,
        )

    def _segment_next_pid(self, current_pid: int) -> int:
        """Return the next policy id to run after the current segment ends.

        Priority:
          1. Head of the Shazam controller's pending queue **only when it points to a
             different policy** (different song / Walk -> mimic). The hit is consumed
             here and used to switch.
          2. **Same-song / empty queue:** keep the current policy. The motion will be
             re-anchored by the next phase seek (or looped if it hit ``MOTION_DONE``).
             We deliberately do NOT fall back to Stand mid-song: that was producing
             a Stand+priming gap of ~2s every 5s, making the robot appear one chunk
             ahead of the music.
        """
        self._drop_stale_hits()
        hit = self._peek_shazam_hit()
        if hit is not None and int(getattr(hit, "policy_id", -1)) != current_pid:
            popped = self._pop_shazam_hit()
            if popped is not None:
                self._last_applied_hit = popped
                self._current_song_id = getattr(popped, "song_id", None)
                return int(popped.policy_id)
        # Same-song hit (don't consume here, Step 1 will phase-seek it next step) or
        # empty queue: stay on the current BeyondMimic so the music keeps driving it.
        return current_pid

    def _consume_sim_reborn(self, ctrl_data) -> bool:
        """R / [SIM_REBORN]: switch to locomotion (Walk) first, then reborn — before get_observation."""
        cmds = ctrl_data.get("COMMANDS", [])
        if "[SIM_REBORN]" not in cmds or not hasattr(self.env, "reborn"):
            return False

        walk_id = int(self.cfg.sim_reborn_locomotion_policy_id)
        if not (0 <= walk_id < self.policy_manager.num_policies):
            walk_id = 0

        self.policy_manager.timer.clear()
        self._clear_beyondmimic_prime()
        self._motion_done_switch_queued = False
        self.policy_manager.warmup_policy_indices.clear()

        if self.policy_manager.current_policy_id != walk_id:
            logger.warning(f"[SIM_REBORN] Switch to Walk (policy {walk_id}), then reborn.")
            self.policy_manager.set_policy(walk_id)
        else:
            logger.warning("[SIM_REBORN] Already on Walk — reborn.")

        logger.warning("Simulation Env reborn!")
        self.env.reborn()  # pyright: ignore[reportAttributeAccessIssue]
        self.timestep = 0
        self.policy.reset_alignment()
        self.policy.reset()
        self.policy_manager.warmup_policy_indices.clear()
        self.ctrl_manager.reset()
        self.policy_manager.timer.clear()
        ctrl_data["COMMANDS"] = [c for c in cmds if c != "[SIM_REBORN]"]
        return True

    def _neutralize_ctrl_for_stand_prime(self, ctrl_data) -> None:
        """Zero locomotion inputs while Stand is holding before a BeyondMimic switch."""
        if self._pending_beyondmimic_id is None or self._stand_policy_id is None:
            return
        if self.policy_manager.current_policy_id != self._stand_policy_id:
            return

        kb = ctrl_data.get("KeyboardCtrl")
        if kb is not None and "keyboard_event" in kb:
            kb["keyboard_event"] = []

        js = ctrl_data.get("JoystickCtrl")
        if js is not None:
            axes = js.get("axes")
            if axes is not None:
                for key in list(axes.keys()):
                    axes[key] = 0.0

        unitree = ctrl_data.get("UnitreeCtrl")
        if unitree is not None:
            axes = unitree.get("axes")
            if axes is not None:
                for key in list(axes.keys()):
                    axes[key] = 0.0

    def post_step_callback(self, env_data, ctrl_data, extras, pd_target):
        self.timestep += 1

        commands = ctrl_data.get("COMMANDS", [])
        commands = self._expand_listener_commands(commands)
        self._maybe_apply_remote_motion_timer(commands)

        # Reset step-scoped arbitration flags. Priority order for automatic
        # switches within one step (Shazam-first):
        #   1) Shazam head (same-song -> seek; different-song -> switch)
        #   2) MOTION_DONE callback
        #   3) Per-policy segment time-box
        # Manual/user commands (keyboard, joystick, SIM_REBORN, SHUTDOWN) that
        # come in via ``commands`` are always honored by the dispatcher below.
        shazam_applied_this_step: bool = False
        self._boundary_consumed_this_step = False

        cur_pid = self.policy_manager.current_policy_id

        # When we're not on a BeyondMimic policy (Walk/Stand), forget the current
        # song so the next Shazam hit triggers a fresh switch rather than a seek.
        if not self._is_beyondmimic_policy_id(cur_pid):
            self._current_song_id = None

        # If we've actually landed on a BeyondMimic policy (either the one we
        # were priming for, or because Shazam pushed us forward), the priming is
        # done. Without this, the Stand-prime gate stays armed forever and Step 1
        # / Step 3 are blocked: every subsequent Shazam chunk piles up in the
        # controller's bounded deque and is silently dropped.
        if (
            self._pending_beyondmimic_id is not None
            and self._is_beyondmimic_policy_id(cur_pid)
        ):
            logger.info(
                "Landed on BeyondMimic policy %d -> clearing pending prime (was %s).",
                cur_pid,
                self._pending_beyondmimic_id,
            )
            self._clear_beyondmimic_prime()

        if self._pending_safe_window_policy_id is not None and cur_pid == self._pending_safe_window_policy_id:
            self._apply_safe_window(cur_pid)
            self._pending_safe_window_policy_id = None

        # --- Step 0: drain stale / same-target hits during Stand priming ---
        # If we're still priming Stand toward a target BeyondMimic, drop chunks
        # that match the same target (no need to re-issue the same switch) and
        # let any *other* song wait its turn (it will be handled the next step
        # once `_pending_beyondmimic_id` is cleared after we land).
        if self._pending_beyondmimic_id is not None:
            self._drop_stale_hits()
            target_pid = self._pending_beyondmimic_id
            while True:
                head = self._peek_shazam_hit()
                if head is None:
                    break
                head_pid = int(getattr(head, "policy_id", -1))
                if head_pid != target_pid:
                    break
                self._pop_shazam_hit()  # silently consume "I want X" while we're already going to X.

        # --- Step 1: Shazam-first arbitration -------------------------------
        # Skip while Stand priming is in flight: we don't want to fight the
        # priming state machine mid-transition.
        if self._pending_beyondmimic_id is None:
            self._drop_stale_hits()
            hit = self._peek_shazam_hit()
            if hit is not None:
                hit_song = getattr(hit, "song_id", None)
                hit_pid = int(getattr(hit, "policy_id", -1))
                same_song = (
                    self._current_song_id is not None
                    and hit_song is not None
                    and hit_song == self._current_song_id
                    and self._is_beyondmimic_policy_id(cur_pid)
                )
                if same_song:
                    popped = self._pop_shazam_hit()
                    if popped is not None:
                        self._apply_phase_seek(popped)
                        self._last_applied_hit = popped
                        self._current_song_id = getattr(popped, "song_id", None)
                        shazam_applied_this_step = True
                        # If the motion had finished naturally, loop it so the
                        # robot keeps dancing while the song is still playing.
                        inner = getattr(self.policy, "policy", None)
                        if inner is not None and getattr(inner, "flag_motion_done", False):
                            if hasattr(inner, "reset"):
                                inner.reset()
                            if hasattr(inner, "play_speed_default"):
                                inner.play_speed = float(getattr(inner, "play_speed_default", 1.0))
                            if hasattr(inner, "flag_motion_done"):
                                inner.flag_motion_done = False
                            logger.info(
                                "Motion done while song %s active -> looping policy %d.",
                                self._current_song_id,
                                cur_pid,
                            )
                        # Reset the time-box watchdog: as long as same-song chunks
                        # keep arriving, we never trip the auto-Stand fallback.
                        self._segment_t0_mono = time.monotonic()
                        self._segment_switch_fired = False
                        logger.info(
                            "Shazam same-song chunk consumed (policy %d, no switch).",
                            cur_pid,
                        )
                elif (
                    0 <= hit_pid < self.policy_manager.num_policies
                    and hit_pid != cur_pid
                ):
                    popped = self._pop_shazam_hit()
                    if popped is not None:
                        commands.append(f"[POLICY_SWITCH],{hit_pid}")
                        self._last_applied_hit = popped
                        self._current_song_id = getattr(popped, "song_id", None)
                        shazam_applied_this_step = True
                        self._boundary_consumed_this_step = True
                        logger.info(
                            "Shazam queue -> switching policy %d -> %d (song=%s).",
                            cur_pid,
                            hit_pid,
                            hit_song,
                        )

        # --- Step 2: MOTION_DONE callback -----------------------------------
        # Only honor MOTION_DONE when Shazam hasn't claimed the step and no
        # other boundary-switch was already queued.
        for callback in extras.get("CALLBACK", []) or []:
            if callback != "[MOTION_DONE]":
                continue
            if shazam_applied_this_step or self._boundary_consumed_this_step:
                logger.info("Shazam override: skipping MOTION_DONE this step.")
                continue
            # If a song is still being identified by Shazam, loop the motion
            # in place instead of falling back to Walk (which created a
            # noticeable gap mid-song).
            if (
                self._is_beyondmimic_policy_id(cur_pid)
                and self._current_song_id is not None
                and not self._is_current_song_stale()
            ):
                inner = getattr(self.policy, "policy", None)
                if inner is not None:
                    if hasattr(inner, "reset"):
                        inner.reset()
                    if hasattr(inner, "play_speed_default"):
                        inner.play_speed = float(getattr(inner, "play_speed_default", 1.0))
                    if hasattr(inner, "flag_motion_done"):
                        inner.flag_motion_done = False
                self._segment_t0_mono = time.monotonic()
                self._segment_switch_fired = False
                logger.info(
                    "Motion done -> looping policy %d (song=%s still active).",
                    cur_pid,
                    self._current_song_id,
                )
                continue
            fallback_pid = int(getattr(self.cfg, "motion_done_fallback_policy_id", 0))
            if not (0 <= fallback_pid < self.policy_manager.num_policies):
                fallback_pid = 0
            if self.policy_manager.current_policy_id != fallback_pid and not self._motion_done_switch_queued:
                self._motion_done_switch_queued = True
                self._boundary_consumed_this_step = True
                commands.append(f"[POLICY_SWITCH],{fallback_pid}")
                logger.info("Motion done -> auto switch to policy %d.", fallback_pid)

        # --- Step 3: Per-policy segment time-box ----------------------------
        if (
            not shazam_applied_this_step
            and not self._boundary_consumed_this_step
            and not self._segment_switch_fired
            and self._segment_pid == cur_pid
            and cur_pid not in self.cfg.policy_inference_timing_skip_policy_ids
            and self._pending_beyondmimic_id is None
        ):
            dur = self._segment_duration_by_pid.get(cur_pid, 0.0)
            if dur > 0.0 and (time.monotonic() - self._segment_t0_mono) >= dur:
                next_pid = self._segment_next_pid(cur_pid)
                if next_pid != cur_pid:
                    commands.append(f"[POLICY_SWITCH],{next_pid}")
                    self._segment_switch_fired = True
                    self._boundary_consumed_this_step = True
                    logger.info(
                        "Segment time-box elapsed (%.3fs) for policy %d %s -> auto switch to %d.",
                        dur,
                        cur_pid,
                        self.policy.name,
                        next_pid,
                    )

        for command in commands:
            match command:
                case "[SHUTDOWN]":
                    logger.warning("Emergency shutdown!")
                    self.env.shutdown()
                case "[SAFE_SNAPSHOT_SAVE]":
                    self._save_safe_snapshot()
                case cmd if cmd.startswith("[SAFE_WINDOW]"):
                    policy_id = int(cmd.split(",")[1])
                    if policy_id < self.policy_manager.num_policies:
                        self._pending_safe_window_policy_id = policy_id
                        if self.policy_manager.current_policy_id == policy_id:
                            self._apply_safe_window(policy_id)
                            self._pending_safe_window_policy_id = None
                        else:
                            self._request_policy_switch(policy_id, skip_stand_prime=True)
                case "[POLICY_TOGGLE]":
                    logger.warning("Policy toggled!")
                    next_policy_id = (self.policy_manager.current_policy_id + 1) % self.policy_manager.num_policies
                    self._request_policy_switch(next_policy_id)

                case cmd if cmd.startswith("[POLICY_SWITCH_DIRECT]"):
                    policy_id = int(cmd.split(",")[1])
                    if policy_id < self.policy_manager.num_policies:
                        self._request_policy_switch(policy_id, skip_stand_prime=True)

                case cmd if cmd.startswith("[POLICY_SWITCH]"):
                    policy_id = int(cmd.split(",")[1])
                    if policy_id < self.policy_manager.num_policies:
                        self._request_policy_switch(policy_id)

        self.ctrl_manager.post_step_callback(ctrl_data)

        self.policy.post_step_callback(commands)
        if self.visualizer is not None:
            self.policy.debug_viz(self.visualizer, env_data, ctrl_data, extras)

        self.policy_manager.step(env_data, ctrl_data)

        # Finish Stand -> BeyondMimic priming once stand is active and delayed switches drained.
        if self._pending_beyondmimic_id is not None and self._stand_policy_id is not None:
            stand_id = self._stand_policy_id
            target_id = self._pending_beyondmimic_id
            if self.policy_manager.current_policy_id == stand_id and not self.policy_manager.timer.has_pending():
                if self._beyondmimic_prime_steps_remaining <= 0:
                    self._beyondmimic_prime_steps_remaining = self._beyondmimic_stand_hold_steps()
                self._beyondmimic_prime_steps_remaining -= 1
                if self._beyondmimic_prime_steps_remaining <= 0:
                    logger.info(f"Stand priming done -> switching to BeyondMimic policy {target_id}.")
                    self._reset_policy_runtime_overrides(target_id)
                    self.policy_manager.switch_policy(target_id, fast=True)
                    self._clear_beyondmimic_prime()

        # Once we're back to policy 0, allow auto-switch to trigger again
        # for the next mimic run.
        fallback_pid = int(getattr(self.cfg, "motion_done_fallback_policy_id", 0))
        if self.policy_manager.current_policy_id == fallback_pid:
            self._motion_done_switch_queued = False

        self._update_mujoco_viewer_progress()

        self.safety_check()
        if self.cfg.debug.log_obs:
            self.debug_logger.log(
                env_data=env_data,
                ctrl_data=ctrl_data,
                extras=extras,
                pd_target=pd_target,
                timestep=self.timestep,
            )

    def step(self, dry_run=False):
        self.env.update()
        env_data = self.env.get_data()
        ctrl_data = self.ctrl_manager.get_ctrl_data(env_data)

        if self._consume_sim_reborn(ctrl_data):
            self.env.update()
            env_data = self.env.get_data()

        commands = ctrl_data.get("COMMANDS", [])
        if len(commands) > 0:
            logger.info(f"{'=' * 10} COMMANDS {'=' * 10}\n{commands}")

        pid = self.policy_manager.current_policy_id
        self._log_policy_segment_wallclock(pid, self.cfg.policy_inference_timing_skip_policy_ids)

        self._neutralize_ctrl_for_stand_prime(ctrl_data)

        obs, extras = self.policy.get_observation(env_data, ctrl_data)
        pd_target = self.policy.get_pd_target(obs)

        if not dry_run:
            self.env.step(pd_target, extras.get("hand_pose", None))
            # logger.debug(pd_target)

        self.post_step_callback(env_data, ctrl_data, extras, pd_target)


if __name__ == "__main__":
    pass
