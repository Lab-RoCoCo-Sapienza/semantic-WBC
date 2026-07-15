import os

from robojudo.config import cfg_registry
from robojudo.controller.ctrl_cfgs import (
    JoystickCtrlCfg,  # noqa: F401
    KeyboardCtrlCfg,  # noqa: F401
    MusicCommandSocketClientCtrlCfg,  # noqa: F401
    MusicCommandSocketCtrlCfg,  # noqa: F401
    PolicyGuiCtrlCfg,  # noqa: F401
    ShazamFileWatchCtrlCfg,  # noqa: F401
    UnitreeCtrlCfg,  # noqa: F401
)
from robojudo.pipeline.pipeline_cfgs import (
    RlLocoMimicPipelineCfg,  # noqa: F401
    RlMultiPolicyPipelineCfg,  # noqa: F401
    RlPipelineCfg,  # noqa: F401
)

from .ctrl.g1_beyondmimic_ctrl_cfg import G1BeyondmimicCtrlCfg  # noqa: F401
from .ctrl.g1_motion_ctrl_cfg import (  # noqa: F401
    G1MotionCtrlCfg,
    G1MotionH2HCtrlCfg,
    G1MotionKungfuBotCtrlCfg,
    G1MotionTwistCtrlCfg,
)
from .ctrl.g1_twist_redis_ctrl_cfg import G1TwistRedisCtrlCfg  # noqa: F401
from .env.g1_dummy_env_cfg import G1DummyEnvCfg  # noqa: F401
from .env.g1_mujuco_env_cfg import G1_12MujocoEnvCfg, G1_23MujocoEnvCfg, G1MujocoEnvCfg  # noqa: F401
from .env.g1_real_env_cfg import G1RealEnvCfg, G1UnitreeCfg  # noqa: F401
from .policy.g1_amo_policy_cfg import G1AmoPolicyCfg  # noqa: F401
from .policy.g1_asap_policy_cfg import G1AsapLocoPolicyCfg, G1AsapPolicyCfg  # noqa: F401
from .policy.g1_beyondmimic_policy_cfg import G1BeyondMimicPolicyCfg  # noqa: F401
from .policy.g1_h2h_policy_cfg import G1H2HPolicyCfg  # noqa: F401
from .policy.g1_kungfubot_policy_cfg import G1KungfuBotGeneralPolicyCfg, G1KungfuBotPolicyCfg  # noqa: F401
from .policy.g1_protomotions_tracker_cfg import ProtoMotionsTrackerPolicyCfg  # noqa: F401
from .policy.g1_smooth_policy_cfg import G1SmoothPolicyCfg  # noqa: F401
from .policy.g1_twist_policy_cfg import G1TwistPolicyCfg  # noqa: F401
from .policy.g1_unitree_policy_cfg import G1UnitreePolicyCfg, G1UnitreeWoGaitPolicyCfg  # noqa: F401


# ======================== Basic Configs ======================== #
@cfg_registry.register
class g1(RlPipelineCfg):
    """
    Unitree G1 robot configuration, Unitree Policy, Sim2Sim.
    You can modify to play with other policies and controllers.
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg()
    # env: G1_23MujocoEnvCfg = G1_23MujocoEnvCfg()
    # env: G1_12MujocoEnvCfg = G1_12MujocoEnvCfg()

    ctrl: list[JoystickCtrlCfg | KeyboardCtrlCfg] = [  # note: the ranking of controllers matters
        JoystickCtrlCfg(),
        # KeyboardCtrlCfg(),
    ]

    policy: G1UnitreePolicyCfg = G1UnitreePolicyCfg()
    # policy: G1UnitreeWoGaitPolicyCfg = G1UnitreeWoGaitPolicyCfg()
    # policy: G1AmoPolicyCfg = G1AmoPolicyCfg()

    # run_fullspeed: bool = env.is_sim


@cfg_registry.register
class g1_real(g1):
    """
    Unitree G1 robot, Unitree Policy, Sim2Real.
    To extend the sim2sim config to sim2real, just need to change the env to real env.
    """

    # env: G1DummyEnvCfg = G1DummyEnvCfg()
    env: G1RealEnvCfg = G1RealEnvCfg(
        # env_type="UnitreeEnv",  # For unitree_sdk2py
        env_type="UnitreeCppEnv",  # For unitree_cpp, check README for more details
        unitree=G1UnitreeCfg(
            net_if="eth0",  # note: change to your network interface
        ),
    )

    ctrl: list[UnitreeCtrlCfg] = [
        UnitreeCtrlCfg(),
    ]

    do_safety_check: bool = True  # enable safety check for real robot


@cfg_registry.register
class g1_switch(RlMultiPolicyPipelineCfg):
    """
    Example of multi-policy pipeline configuration.
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg()

    ctrl: list[KeyboardCtrlCfg | JoystickCtrlCfg] = [
        # KeyboardCtrlCfg(
        #     triggers_extra={
        #         "Key.tab": "[POLICY_TOGGLE]",
        #     }
        # ),
        JoystickCtrlCfg(
            triggers_extra={
                "RB+Down": "[POLICY_SWITCH],0",
                "RB+Up": "[POLICY_SWITCH],1",
            }
        ),
    ]

    policies: list[G1UnitreePolicyCfg | G1AmoPolicyCfg] = [
        G1UnitreePolicyCfg(),
        G1AmoPolicyCfg(),
    ]


@cfg_registry.register
class g1_locomimic(RlLocoMimicPipelineCfg):
    """
    Example of loco mimic pipeline configuration.
    You can switch between loco and mimic policies during runtime, with interpolation.
    === Check more fancy locomimic examples in g1_loco_mimic_cfg.py ===
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg()

    ctrl: list[KeyboardCtrlCfg | JoystickCtrlCfg] = [
        KeyboardCtrlCfg(
            triggers_extra={
                "]": "[POLICY_LOCO]",
                "[": "[POLICY_MIMIC]",
            }
        ),
        JoystickCtrlCfg(
            triggers_extra={
                "RB+Down": "[POLICY_LOCO]",
                "RB+Up": "[POLICY_MIMIC]",
            }
        ),
    ]

    loco_policy: G1UnitreePolicyCfg = G1UnitreePolicyCfg()
    mimic_policies: list[G1AsapPolicyCfg] = [
        G1AsapPolicyCfg(),
    ]


# ======================== Configs for supported Policy ======================== #


@cfg_registry.register
class g1_h2h(RlPipelineCfg):
    """
    Human2Humanoid
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg()
    ctrl: list[KeyboardCtrlCfg | G1MotionH2HCtrlCfg] = [
        KeyboardCtrlCfg(),
        G1MotionH2HCtrlCfg(),
    ]

    policy: G1H2HPolicyCfg = G1H2HPolicyCfg()


@cfg_registry.register
class g1_beyondmimic(RlPipelineCfg):
    """
    BeyondMimic Policy, support both with and without state estimator.
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg()
    ctrl: list[KeyboardCtrlCfg] = [
        KeyboardCtrlCfg(),
    ]

    policy: G1BeyondMimicPolicyCfg = G1BeyondMimicPolicyCfg(
        policy_name="Jump_wose",
        without_state_estimator=True,
        use_modelmeta_config=True,  # use robot dof config from modelmeta
        use_motion_from_model=True,  # use motion from onnx model
        max_timestep=140,
    )


@cfg_registry.register
class g1_beyondmimic_with_ctrl(RlPipelineCfg):
    """
    BeyondMimic with External BeyondMimicCtrl as motion source.
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg()
    ctrl: list[KeyboardCtrlCfg | G1BeyondmimicCtrlCfg] = [
        KeyboardCtrlCfg(),
        G1BeyondmimicCtrlCfg(
            motion_name="dance1_subject2",  # you can put your own motion file in assets/motions/g1
        ),
    ]

    policy: G1BeyondMimicPolicyCfg = G1BeyondMimicPolicyCfg(
        policy_name="Dance_wose",
        use_motion_from_model=False,  # use motion from BeyondmimicCtrl instead of the onnx
    )


@cfg_registry.register
class g1_asap(RlPipelineCfg):
    """
    Unitree G1 robot configuration, ASAP Policy, Sim2Sim.
    You can modify to play with other policies and controllers.
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg(forward_kinematic=None, update_with_fk=False, born_place_align=True)

    ctrl: list[JoystickCtrlCfg | KeyboardCtrlCfg] = [  # note: the ranking of controllers matters
        # JoystickCtrlCfg(),
        KeyboardCtrlCfg(triggers={"r": "[SIM_REBORN]", "o": "[SHUTDOWN]", "|": "[MOTION_RESET]"}),
    ]

    policy: G1AsapPolicyCfg = G1AsapPolicyCfg()
    """You can also try other models, from ASAP, RoboMimic, KungfuBot(PBHC)"""
    # policy: G1KungfuBotPolicyCfg = G1KungfuBotPolicyCfg() # KungfuBot horse_squat
    # # fmt: off
    # policy: G1AsapPolicyCfg = G1AsapPolicyCfg(
    #     policy_name="robomimic",
    #     relative_path="dance_0605.onnx",
    #     motion_length_s=18.0,
    #     start_upper_body_dof_pos = [
    #         0, 0, 0,
    #         0.35, 0.18, 0, 0.87,
    #         0.35, -0.18, 0, 0.87,
    #     ],
    # )
    # # fmt: on


@cfg_registry.register
class g1_asap_loco(RlPipelineCfg):
    """
    Unitree G1 robot configuration, ASAP Locomotion Policy, Sim2Sim.
    You can modify to play with other policies and controllers.
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg(forward_kinematic=None, update_with_fk=False, born_place_align=False)

    ctrl: list[JoystickCtrlCfg | KeyboardCtrlCfg] = [  # note: the ranking of controllers matters
        # JoystickCtrlCfg(),
        KeyboardCtrlCfg(
            triggers={
                "i": "[SIM_REBORN]",
                "o": "[SHUTDOWN]",
            }
        ),
    ]

    policy: G1AsapLocoPolicyCfg = G1AsapLocoPolicyCfg()


@cfg_registry.register
class g1_kungfubot2(RlPipelineCfg):
    """
    PBHC KungfuBot2 General Policy
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg()
    ctrl: list[KeyboardCtrlCfg | G1MotionKungfuBotCtrlCfg] = [
        KeyboardCtrlCfg(),
        G1MotionKungfuBotCtrlCfg(
            motion_name="kungfubot/Horse-stance_pose",  # put motion files in assets/motions/g1/phc/kungfubot
        ),
    ]

    policy: G1KungfuBotGeneralPolicyCfg = G1KungfuBotGeneralPolicyCfg(
        policy_name="horse_test_43000",  # this is a test model trained with only one motion
        compatibility_old_version=True,  # for old version of kungfubot general policy (before 2025-11-13 bugfix #68)
    )


@cfg_registry.register
class g1_twist(RlPipelineCfg):
    """
    Unitree G1 robot configuration, TWIST Policy, Sim2Sim.
    TwistRedisCtrl for the original repo of high level motion stream over redis.
    MotionTwistCtrl for built-in motion control.
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg(forward_kinematic=None, update_with_fk=False, born_place_align=False)

    ctrl: list[G1TwistRedisCtrlCfg | G1MotionTwistCtrlCfg] = [  # note: the ranking of controllers matters
        G1TwistRedisCtrlCfg(redis_host="localhost"),  # with hign level motion lib through redis
        # G1MotionTwistCtrlCfg(), # with built-in motion ctrl
    ]

    policy: G1TwistPolicyCfg = G1TwistPolicyCfg()


# ======================== Fancy Example Configs ======================== #


@cfg_registry.register
class g1_switch_beyondmimic(RlMultiPolicyPipelineCfg):
    """
    Switch between multiple BeyondMimic policies. Withour Interpolation.
    """

    policy_inference_log_interval_s: float = 1.0

    # MuJoCo HUD hotkey strip (right column only; see mujoco_viewer_progress._merge_legend_rows).
    mujoco_overlay_legend: list[tuple[str, str]] = [
        (
            "",
            "w Walk | s Stand | 1 bts2 | 2 dynamite | 3 easy | 4 swim | 5 thriller | 6 salsa | 7 gdance | 8 salsa4 | 9 thrillerLW | Tab next | r reborn | Esc quit",
        ),
    ]

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg()
    ctrl: list[KeyboardCtrlCfg | JoystickCtrlCfg] = [
        KeyboardCtrlCfg(
            triggers_extra={
                "Key.tab": "[POLICY_TOGGLE]",
                # Unitree modes
                "w": "[POLICY_SWITCH],0",  # Walk (initial)
                "s": "[POLICY_SWITCH],1",  # Stand (no gait)
                # BeyondMimic models
                "1": "[POLICY_SWITCH],2",
                "2": "[POLICY_SWITCH],3",
                "3": "[POLICY_SWITCH],4",
                "4": "[POLICY_SWITCH],5",
                "5": "[POLICY_SWITCH],6",
                "6": "[POLICY_SWITCH],7",
                "7": "[POLICY_SWITCH],8",
                "8": "[POLICY_SWITCH],9",
                "9": "[POLICY_SWITCH],10",
                "!": "[POLICY_SWITCH],2",  # note: with shift
                "@": "[POLICY_SWITCH],3",  # note: with shift
                "#": "[POLICY_SWITCH],4",  # note: with shift
                "$": "[POLICY_SWITCH],5",  # note: with shift
                "%": "[POLICY_SWITCH],6",  # shift+5 (US)
                "^": "[POLICY_SWITCH],7",  # shift+6 (US)
                "&": "[POLICY_SWITCH],8",  # shift+7 (US)
                "*": "[POLICY_SWITCH],9",  # shift+8 (US)
                "(": "[POLICY_SWITCH],10",  # shift+9 (US)
            }
        ),
        JoystickCtrlCfg(
            triggers_extra={
                # Unitree modes
                "RB+Down": "[POLICY_SWITCH],0",
                "RB+Left": "[POLICY_SWITCH],1",
                # BeyondMimic models
                "RB+Up": "[POLICY_SWITCH],2",
                "RB+Right": "[POLICY_SWITCH],3",
            }
        ),
    ]

    # Policy list order defines the initial policy (index 0).
    policies: list[G1UnitreePolicyCfg | G1UnitreeWoGaitPolicyCfg | G1BeyondMimicPolicyCfg] = [
        # 0: Unitree Walk
        G1UnitreePolicyCfg(),
        # 1: Unitree Stand (no gait / no stepping when standing)
        G1UnitreeWoGaitPolicyCfg(),
        # BeyondMimic (auto-curated from assets/models/g1/beyondmimic/*.onnx)
        # NOTE on play_speed / segment_ratio:
        #   * ``play_speed`` is the MOTION speedup (timestep += play_speed each env
        #     step). Higher = motion plays faster / shorter, so it stays inside one
        #     5s audio chunk and feels "in time".
        #   * ``segment_ratio`` is a wall-clock watchdog multiplier on the time-box
        #     (5s default). Kept at 1.0 so the watchdog matches one chunk: it only
        #     trips if Shazam stops sending hits for that song.
        G1BeyondMimicPolicyCfg(
            policy_name="bts_2_0_tracking_2",
            without_state_estimator=False,
            use_modelmeta_config=True,
            use_motion_from_model=True,
            max_timestep=-1,
            auto_motion_length=True,
            play_speed=1.0,
        ),
        G1BeyondMimicPolicyCfg(
            policy_name="bts_dynamite_tracking",
            without_state_estimator=False,
            use_modelmeta_config=True,
            use_motion_from_model=True,
            max_timestep=-1,
            auto_motion_length=True,
            play_speed=1.0,
        ),
        G1BeyondMimicPolicyCfg(
            policy_name="easy_sample",
            without_state_estimator=False,
            use_modelmeta_config=True,
            use_motion_from_model=True,
            max_timestep=-1,
            auto_motion_length=True,
            play_speed=1.0,
        ),
        G1BeyondMimicPolicyCfg(
            policy_name="Swim_tracking",
            without_state_estimator=False,
            use_modelmeta_config=True,
            use_motion_from_model=True,
            max_timestep=-1,
            auto_motion_length=True,
            play_speed=1.0,
        ),
        G1BeyondMimicPolicyCfg(
            policy_name="thriller",
            without_state_estimator=False,
            use_modelmeta_config=True,
            use_motion_from_model=True,
            max_timestep=-1,
            auto_motion_length=True,
            play_speed=1.0,
        ),
        G1BeyondMimicPolicyCfg(
            policy_name="salsa_tracking",
            without_state_estimator=False,
            use_modelmeta_config=True,
            use_motion_from_model=True,
            max_timestep=-1,
            auto_motion_length=True,
            play_speed=1.0,
        ),
        G1BeyondMimicPolicyCfg(
            policy_name="gdance",
            without_state_estimator=False,
            use_modelmeta_config=True,
            use_motion_from_model=True,
            max_timestep=-1,
            auto_motion_length=True,
            play_speed=1.0,
        ),
        G1BeyondMimicPolicyCfg(
            policy_name="Salsa_4",
            without_state_estimator=False,
            use_modelmeta_config=True,
            use_motion_from_model=True,
            max_timestep=-1,
            auto_motion_length=True,
            play_speed=1.0,
            locked_joint_names=[
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint",
            ],
        ),
        G1BeyondMimicPolicyCfg(
            policy_name="thriller_locked_waist",
            without_state_estimator=False,
            use_modelmeta_config=True,
            use_motion_from_model=True,
            max_timestep=-1,
            auto_motion_length=True,
            play_speed=1.0,
        ),
    ]


@cfg_registry.register
class g1_switch_beyondmimic_gesti_safe(RlMultiPolicyPipelineCfg):
    """Switch config with selected BeyondMimic ONNX + SAFE snapshot hotkey."""

    policy_inference_log_interval_s: float = 1.0
    beyondmimic_skip_stand_prime: bool = True
    safe_policy_windows: dict[int, tuple[int, int]] = {
        2: (0, 600),        # gesti safe steps
        3: (1146, 1549),    # 67 safe steps
    }
    mujoco_overlay_legend: list[tuple[str, str]] = [
        ("", "w Walk | s Stand | 1 gesti | 2 67 | 3 salsa | 4 thriller | 5 67_safe_onnx | 6 gesti_safe_onnx | i salva SAFE step | Tab next | r reborn | Esc quit"),
    ]

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg()

    ctrl: list[KeyboardCtrlCfg | JoystickCtrlCfg] = [
        KeyboardCtrlCfg(
            triggers_extra={
                "Key.tab": "[POLICY_TOGGLE]",
                "w": "[POLICY_SWITCH],0",
                "s": "[POLICY_SWITCH],1",
                "i": "[SAFE_SNAPSHOT_SAVE]",
                "1": "[POLICY_SWITCH],2",
                "2": "[POLICY_SWITCH],3",
                "3": "[POLICY_SWITCH],4",
                "4": "[POLICY_SWITCH],5",
                "5": "[POLICY_SWITCH],6",
                "6": "[POLICY_SWITCH],7",
                "g": "[POLICY_SWITCH],6",
                "h": "[POLICY_SWITCH],7",
            }
        ),
        JoystickCtrlCfg(
            triggers_extra={
                "RB+Down": "[POLICY_SWITCH],0",
                "RB+Left": "[POLICY_SWITCH],1",
                "RB+Up": "[POLICY_SWITCH],2",
                "RB+Right": "[POLICY_SWITCH],3",
            }
        ),
    ]

    policies: list[G1UnitreePolicyCfg | G1UnitreeWoGaitPolicyCfg | G1BeyondMimicPolicyCfg] = [
        G1UnitreePolicyCfg(),
        G1UnitreeWoGaitPolicyCfg(),
        G1BeyondMimicPolicyCfg(
            policy_name="g1_29dof_gesti_10k",
            without_state_estimator=False,
            use_modelmeta_config=True,
            use_motion_from_model=True,
            max_timestep=-1,
            auto_motion_length=True,
            play_speed=1.0,
        ),
        G1BeyondMimicPolicyCfg(
            policy_name="g1_29dof_67_10k",
            without_state_estimator=False,
            use_modelmeta_config=True,
            use_motion_from_model=True,
            max_timestep=-1,
            auto_motion_length=True,
            play_speed=1.0,
        ),
        G1BeyondMimicPolicyCfg(
            policy_name="salsa_tracking",
            without_state_estimator=False,
            use_modelmeta_config=True,
            use_motion_from_model=True,
            max_timestep=-1,
            auto_motion_length=True,
            play_speed=1.84,
        ),
        G1BeyondMimicPolicyCfg(
            policy_name="thriller",
            without_state_estimator=False,
            use_modelmeta_config=True,
            use_motion_from_model=True,
            max_timestep=-1,
            auto_motion_length=True,
            play_speed=1.5,
        ),
        G1BeyondMimicPolicyCfg(
            policy_name="g1_29dof_67_safe",
            without_state_estimator=False,
            use_modelmeta_config=True,
            use_motion_from_model=True,
            max_timestep=-1,
            auto_motion_length=True,
            play_speed=1.0,
        ),
        G1BeyondMimicPolicyCfg(
            policy_name="g1_29dof_gesti_safe",
            without_state_estimator=False,
            use_modelmeta_config=True,
            use_motion_from_model=True,
            max_timestep=-1,
            auto_motion_length=True,
            play_speed=1.0,
        ),
    ]


@cfg_registry.register
class g1_shazam_beyondmimic(g1_switch_beyondmimic):
    """``g1_switch_beyondmimic`` + a file-watch Shazam controller.

    Drop rolling audio chunks (e.g. 10s mp3 files) into ``runtime_chunks/`` and
    the corresponding BeyondMimic policy will be queued and auto-executed for
    ``policy.segment_ratio * segment_beat_s`` seconds. Keyboard/joystick manual
    switching is preserved.
    """

    # Watchdog beat = chunk size; per-policy watchdog = segment_ratio * segment_beat_s.
    # Keep this aligned with the streamer's --segment-seconds.
    segment_beat_s: float = 10.0

    # No Walk → Stand hold before BeyondMimic (see RlMultiPolicyPipelineCfg).
    beyondmimic_skip_stand_prime: bool = True

    ctrl: list[KeyboardCtrlCfg | JoystickCtrlCfg | ShazamFileWatchCtrlCfg] = [
        KeyboardCtrlCfg(
            triggers_extra={
                "Key.tab": "[POLICY_TOGGLE]",
                "w": "[POLICY_SWITCH],0",
                "s": "[POLICY_SWITCH],1",
                "1": "[POLICY_SWITCH],2",
                "2": "[POLICY_SWITCH],3",
                "3": "[POLICY_SWITCH],4",
                "4": "[POLICY_SWITCH],5",
                "5": "[POLICY_SWITCH],6",
                "6": "[POLICY_SWITCH],7",
                "7": "[POLICY_SWITCH],8",
                "8": "[POLICY_SWITCH],9",
                "9": "[POLICY_SWITCH],10",
                "!": "[POLICY_SWITCH],2",
                "@": "[POLICY_SWITCH],3",
                "#": "[POLICY_SWITCH],4",
                "$": "[POLICY_SWITCH],5",
                "%": "[POLICY_SWITCH],6",
                "^": "[POLICY_SWITCH],7",
                "&": "[POLICY_SWITCH],8",
                "*": "[POLICY_SWITCH],9",
                "(": "[POLICY_SWITCH],10",
            }
        ),
        JoystickCtrlCfg(
            triggers_extra={
                "RB+Down": "[POLICY_SWITCH],0",
                "RB+Left": "[POLICY_SWITCH],1",
                "RB+Up": "[POLICY_SWITCH],2",
                "RB+Right": "[POLICY_SWITCH],3",
            }
        ),
        ShazamFileWatchCtrlCfg(
            watch_dir="runtime_chunks",
            watch_name_pattern=r"^\d{4}\.",
            index_path="shazam/index.pkl",
            clap_index_path="shazam/clap_index.json",
            clap_fallback=True,
            clap_fallback_if_fp_votes_below=50,
            # --- Real-time tuning (low-latency Shazam->policy loop) ---
            # NOTE: tuning assumes 10s chunks (segment_beat_s above).
            poll_interval_s=0.1,        # scan watch_dir 10Hz instead of 4Hz
            queue_maxlen=8,             # survives a Stand-prime + transient prime cycle
            max_chunk_age_s=22.0,       # ~2 chunks of grace, tolerate slow startups
            drop_stale_chunks=True,
            # --- Audio playback compensation ---
            # The streamer drops a chunk on disk *before* `afplay` actually starts
            # playing it (queued behind the previous chunk). Hold each Shazam hit
            # in the controller for ~one chunk so the policy switch happens when
            # the user actually hears the chunk, not when it appears on disk.
            controller_delay_s=1.0,
            # --- Song -> policy mapping ---
            song_to_policy={
                "dynamite": 3,
                "swim": 5,
                "salsa4": 9,
                "salsa": 7,
                "bts": 2,
                "thriller": 6,
                "gdance": 8,
            },
        ),
    ]


@cfg_registry.register
class g1_shazam_remote_listener(g1_switch_beyondmimic):
    """Shazam policy switches pushed from an external TCP listener process.

    Run robot side:
        python scripts/run_pipeline.py -c g1_shazam_remote_listener

    Then run audio side (default: no WAV on disk; add ``--save-wav`` to persist clips):
        python scripts/listen_shazam_and_send.py --client-host <robot-ip> --client-port 8765
    """

    # Remote → Stand → (brief hold) → BeyondMimic.
    beyondmimic_skip_stand_prime: bool = False
    beyondmimic_stand_hold_s: float = 0.5
    initial_policy_id: int = 0
    motion_done_fallback_policy_id: int = 0

    ctrl: list[KeyboardCtrlCfg | JoystickCtrlCfg | MusicCommandSocketCtrlCfg] = [
        KeyboardCtrlCfg(
            triggers_extra={
                "Key.tab": "[POLICY_TOGGLE]",
                "w": "[POLICY_SWITCH],0",
                "s": "[POLICY_SWITCH],0",
                "2": "[POLICY_SWITCH],2",
                "3": "[POLICY_SWITCH],3",
                "4": "[POLICY_SWITCH],4",
                "@": "[POLICY_SWITCH],2",
                "#": "[POLICY_SWITCH],3",
                "$": "[POLICY_SWITCH],4",
            }
        ),
        JoystickCtrlCfg(
            triggers_extra={
                "RB+Down": "[POLICY_SWITCH],0",
                "RB+Up": "[POLICY_SWITCH],2",
                "RB+Right": "[POLICY_SWITCH],3",
            }
        ),
        MusicCommandSocketCtrlCfg(
            host="0.0.0.0",
            port=int(os.environ.get("ROBOJUDO_MUSIC_PORT", "8765")),
            queue_maxlen=32,
        ),
    ]

    policies: list[G1UnitreeWoGaitPolicyCfg | G1BeyondMimicPolicyCfg] = [
        G1UnitreeWoGaitPolicyCfg(),  # 0: Stand
        G1UnitreeWoGaitPolicyCfg(),  # 1: reserved (kept for id alignment)
        G1BeyondMimicPolicyCfg(      # 2: salsa
            policy_name="salsa_tracking",
            without_state_estimator=False,
            use_modelmeta_config=True,
            use_motion_from_model=True,
            max_timestep=-1,
            auto_motion_length=True,
            play_speed=1.0,
        ),
        G1BeyondMimicPolicyCfg(      # 3: dynamite
            policy_name="bts_dynamite_tracking",
            without_state_estimator=False,
            use_modelmeta_config=True,
            use_motion_from_model=True,
            max_timestep=-1,
            auto_motion_length=True,
            play_speed=1.0,
        ),
        G1BeyondMimicPolicyCfg(      # 4: thriller
            policy_name="thriller",
            without_state_estimator=False,
            use_modelmeta_config=True,
            use_motion_from_model=True,
            max_timestep=-1,
            auto_motion_length=True,
            play_speed=1.0,
        ),
    ]


@cfg_registry.register
class g1_shazam_remote_listener_headless(g1_shazam_remote_listener):
    """Headless version of `g1_shazam_remote_listener` for benchmark automation.

    Removes keyboard/joystick controllers to avoid macOS Accessibility prompts and
    noisy event-monitor warnings when running unattended.
    """

    _client_host = str(os.environ.get("ROBOJUDO_CMD_SERVER_HOST", "")).strip()
    _client_port = int(os.environ.get("ROBOJUDO_CMD_SERVER_PORT", "8765"))
    _client_topic = str(os.environ.get("ROBOJUDO_CMD_SUBSCRIBE_TOPIC", "gesture")).strip() or "gesture"

    if _client_host:
        ctrl: list[MusicCommandSocketClientCtrlCfg] = [
            MusicCommandSocketClientCtrlCfg(
                host=_client_host,
                port=_client_port,
                subscribe_topic=_client_topic,
                queue_maxlen=32,
                connect_timeout_s=10.0,
                reconnect_interval_s=0.5,
            ),
        ]
    else:
        ctrl: list[MusicCommandSocketCtrlCfg] = [
            MusicCommandSocketCtrlCfg(
                host="0.0.0.0",
                port=int(os.environ.get("ROBOJUDO_MUSIC_PORT", "8766")),
                queue_maxlen=32,
            ),
        ]


# ======================== ProtoMotions Tracker ======================== #


@cfg_registry.register
class g1_protomotions_tracker(RlPipelineCfg):
    """ProtoMotions tracker with cached 50fps motion.

    Uses the standard RoboJuDo G1 MuJoCo environment with ``born_place_align``
    disabled (our policy handles heading alignment itself).

    Usage::

        cd robojudo && python scripts/run_tracker_pipeline.py \\
            -c g1_protomotions_tracker \\
            --onnx-path /path/to/unified_pipeline.onnx \\
            --motion-path /path/to/motion.motion
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg(born_place_align=False)
    policy: ProtoMotionsTrackerPolicyCfg = ProtoMotionsTrackerPolicyCfg()
    ctrl: list[KeyboardCtrlCfg] = [KeyboardCtrlCfg()]


@cfg_registry.register
class g1_protomotions_tracker_real(g1_protomotions_tracker):
    """ProtoMotions tracker on real G1 hardware.

    Usage::

        cd robojudo && python scripts/run_tracker_pipeline.py \\
            -c g1_protomotions_tracker_real \\
            --onnx-path /path/to/unified_pipeline.onnx \\
            --motion-path /path/to/motion.motion
    """

    env: G1RealEnvCfg = G1RealEnvCfg(
        env_type="UnitreeCppEnv",
        unitree=G1UnitreeCfg(
            net_if="eth0",
        ),
        born_place_align=False,
    )
    ctrl: list[UnitreeCtrlCfg] = [
        UnitreeCtrlCfg(),
    ]
    do_safety_check: bool = True


# TIPS: check g1_loco_mimic_cfg.py for more complex examples
