from typing import Any

from pydantic import model_validator

from robojudo.config import Config
from robojudo.controller import CtrlCfg
from robojudo.environment import EnvCfg
from robojudo.policy import PolicyCfg
from robojudo.tools.debug_log import DebugCfg


class PipelineCfg(Config):
    pipeline_type: str  # name of the pipeline class
    # ===== Pipeline Config =====
    device: str = "cpu"

    debug: DebugCfg = DebugCfg()

    run_fullspeed: bool = False
    """If True, run the pipeline at full speed, ignoring the desired frequency"""

    do_safety_check: bool = False
    """
    If True, perform safety check after each step.
    We recommend enabling this, however if motion is very aggressive, you may disable it.
    """


class RlPipelineCfg(PipelineCfg):
    pipeline_type: str = "RlPipeline"

    # ===== Pipeline Config =====
    robot: str  # robot name, e.g. "g1"

    env: EnvCfg | Any
    ctrl: list[CtrlCfg | Any] = []
    policy: PolicyCfg | Any


class RlMultiPolicyPipelineCfg(PipelineCfg):
    pipeline_type: str = "RlMultiPolicyPipeline"

    # ===== Pipeline Config =====
    robot: str  # robot name, e.g. "g1"

    env: EnvCfg | Any
    ctrl: list[CtrlCfg | Any] = []

    policies: list[PolicyCfg | Any] = []
    """First policy as init, rest as extra policies, can be switched to"""

    beyondmimic_stand_hold_s: float = 2.0
    """Wall-clock seconds to hold Stand before switching to BeyondMimic."""

    beyondmimic_prime_policy_id: int | None = None
    """Optional policy id to use as the intermediate priming policy before a BeyondMimic switch.
    If unset, the pipeline auto-detects the UnitreeWoGaitPolicy as the default stand prime.
    """

    initial_policy_id: int = 0
    """Policy id selected after pipeline reset/startup. Default keeps legacy behavior (Walk=0)."""

    motion_done_fallback_policy_id: int = 0
    """Policy id selected when a motion policy finishes and no song is actively keeping it alive."""

    beyondmimic_skip_stand_prime: bool = False
    """If True, never route Walk/Stand → BeyondMimic through a Stand priming hold.
    Matches ``[POLICY_SWITCH_DIRECT]`` for all BeyondMimic targets (keyboard, Tab, etc.).
    """

    segment_beat_s: float = 5.0
    """Shared beat unit; per-policy segment duration = policy.segment_ratio * segment_beat_s."""

    sim_reborn_locomotion_policy_id: int = 0
    """`[SIM_REBORN]` / R: switch to this policy first (e.g. Unitree Walk), then reborn."""

    policy_inference_log_interval_s: float = 0.0
    """If > 0, while a non-skipped policy is active, log wall-clock progress every this many
    seconds (``now``, ``elapsed_s`` since segment start). Ignored for skipped ids."""

    policy_inference_timing_skip_policy_ids: list[int] = [0, 1]
    """Policy indices excluded from wall-clock segment logs (e.g. walk=0, stand=1)."""

    mujoco_overlay_legend: list[tuple[str, str]] | None = None
    """Optional (left, right) ASCII rows under progress in the MuJoCo viewer HUD."""

    safe_snapshot_path: str = "runtime_chunks_live/safe_steps/latest_safe_step.json"
    """Destination JSON file written when `[SAFE_SNAPSHOT_SAVE]` is triggered."""

    safe_snapshot_keep_history: bool = True
    """If True, keep every SAFE snapshot as an additional timestamped JSON file."""

    safe_policy_windows: dict[int, tuple[int, int]] = {}
    """Optional policy_id -> (start_step, end_step) SAFE windows, applied on demand."""

    listener_token_policy_map: dict[str, int] = {}
    """Map smart-mic listener tokens (e.g. ``Violin``, ``67``) to policy ids."""

    listener_gesti_policy_id: int | None = None
    """Policy id for ``[GESTI,<seconds>]`` speech commands from the smart mic listener."""


class RlLocoMimicPipelineCfg(PipelineCfg):
    pipeline_type: str = "RlLocoMimicPipeline"

    # ===== Pipeline Config =====
    robot: str  # robot name, e.g. "g1"

    env: EnvCfg | Any
    ctrl: list[CtrlCfg | Any] = []

    loco_policy: PolicyCfg | Any
    """LocoMotion policy, as init"""
    mimic_policies: list[PolicyCfg | Any] = []
    """MotionMimic policies, can be switched to"""

    # ===== Upper body override Config =====
    upper_dof_num: int = 0
    upper_dof_pos_default: list[float] | None = []
    """Default positions of the upper body DOFs"""
    upper_dof_override_indices: list[int] | None = []
    """Indices of the upper body DOFs to be overridden"""

    @model_validator(mode="after")
    def check_upper_dof(self):
        if self.upper_dof_pos_default is not None:
            if len(self.upper_dof_pos_default) != self.upper_dof_num:
                raise ValueError(
                    f"Length of upper_dof_pos_default ({len(self.upper_dof_pos_default)}) "
                    f"must be equal to upper_dof_num ({self.upper_dof_num})"
                )
        if self.upper_dof_override_indices is not None:
            for idx in self.upper_dof_override_indices:
                if idx < -self.upper_dof_num or idx >= 0:
                    raise ValueError(
                        f"upper_dof_override_indices contains invalid index {idx}, "
                        f"must be in [-{self.upper_dof_num}, 0)"
                    )

        return self
