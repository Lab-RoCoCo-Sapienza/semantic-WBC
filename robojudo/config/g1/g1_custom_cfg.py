from robojudo.config import cfg_registry
from robojudo.controller.ctrl_cfgs import (
    JoystickCtrlCfg,  # noqa: F401
    KeyboardCtrlCfg,  # noqa: F401
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
from .policy.g1_smooth_policy_cfg import G1SmoothPolicyCfg  # noqa: F401
from .policy.g1_twist_policy_cfg import G1TwistPolicyCfg  # noqa: F401
from .policy.g1_unitree_policy_cfg import G1UnitreePolicyCfg, G1UnitreeWoGaitPolicyCfg  # noqa: F401

# ======================== Custom Configs ======================== #
"""
Add your custom config here.
"""


@cfg_registry.register
class g1_dev(RlPipelineCfg):
    robot: str = "g1"
    env: G1_23MujocoEnvCfg = G1_23MujocoEnvCfg()

    ctrl: list[KeyboardCtrlCfg] = [
        KeyboardCtrlCfg(),
    ]

    policy: G1UnitreePolicyCfg = G1UnitreePolicyCfg()


@cfg_registry.register
class g1_gesti_multi_real(RlMultiPolicyPipelineCfg):
    """
    Unitree G1 real robot, RlMultiPolicyPipeline variant:
    - Policy 0: Unitree Stand (no gait) -- also the "loco" fallback
    - Policy 1: g1_29dof_gesti (BeyondMimic)
    - Policy 2: salsa_tracking (BeyondMimic)
    - Policy 3: bts_dynamite_tracking (BeyondMimic)
    - Policy 4: thriller (BeyondMimic)
    - Policy 5: g1_29dof_67_10k (BeyondMimic)
    - Policy 6: Violin (BeyondMimic)

    Joystick mapping:
      X     -> gesti
      Up    -> salsa
      Left  -> bts_dynamite_tracking
      Right -> thriller
      Down  -> 67
      L1    -> Violin
      B     -> Stand (loco)
      A     -> Shutdown
      Y     -> Motion reset

    Walk is intentionally not part of this config: every transition goes
    through Stand (initial policy and fallback after remote motion timer).

    Remote TCP commands ([GESTI,1.06], [Violin], [67]) go through the same
    [POLICY_SWITCH] Stand-priming path (see ``listener_*`` fields below).
    """

    robot: str = "g1"
    env: G1RealEnvCfg = G1RealEnvCfg(
        env_type="UnitreeCppEnv",
        unitree=G1UnitreeCfg(
            net_if="eth0",
        ),
    )

    beyondmimic_skip_stand_prime: bool = False
    beyondmimic_stand_hold_s: float = 0.5

    listener_token_policy_map: dict[str, int] = {"Violin": 6, "67": 5}
    listener_gesti_policy_id: int = 1

    ctrl: list[UnitreeCtrlCfg] = [
        UnitreeCtrlCfg(
            combination_init_buttons=[],
            triggers={
                "A": "[SHUTDOWN]",
                "B": "[POLICY_SWITCH],0",
                "Y": "[MOTION_RESET]",
                "X": "[POLICY_SWITCH],1",
                "Up": "[POLICY_SWITCH],2",
                "Left": "[POLICY_SWITCH],3",
                "Right": "[POLICY_SWITCH],4",
                "Down": "[POLICY_SWITCH],5",
                "L1": "[POLICY_SWITCH],6",
                "V": "[POLICY_SWITCH],3",
            },
        ),
    ]

    policies: list[G1UnitreeWoGaitPolicyCfg | G1BeyondMimicPolicyCfg] = [
        G1UnitreeWoGaitPolicyCfg(),    # 0: Stand (also fallback "loco")
        G1BeyondMimicPolicyCfg(        # 1: gesti
            policy_name="g1_29dof_gesti",
            without_state_estimator=False,
            use_modelmeta_config=True,
            use_motion_from_model=True,
            max_timestep=-1,
            auto_motion_length=True,
            play_speed=1.0,
        ),
        G1BeyondMimicPolicyCfg(        # 2: salsa
            policy_name="salsa_tracking",
            without_state_estimator=False,
            use_modelmeta_config=True,
            use_motion_from_model=True,
            max_timestep=-1,
            auto_motion_length=True,
            play_speed=1.0,
        ),
        G1BeyondMimicPolicyCfg(        # 3: bts_dynamite_tracking
            policy_name="bts_dynamite_tracking",
            without_state_estimator=False,
            use_modelmeta_config=True,
            use_motion_from_model=True,
            max_timestep=-1,
            auto_motion_length=True,
            play_speed=1.0,
        ),
        G1BeyondMimicPolicyCfg(        # 4: thriller
            policy_name="thriller",
            without_state_estimator=False,
            use_modelmeta_config=True,
            use_motion_from_model=True,
            max_timestep=-1,
            auto_motion_length=True,
            play_speed=0.85,
        ),
        G1BeyondMimicPolicyCfg(        # 5: 67
            policy_name="g1_29dof_67_10k",
            without_state_estimator=False,
            use_modelmeta_config=True,
            use_motion_from_model=True,
            max_timestep=-1,
            auto_motion_length=True,
            play_speed=1.0,
        ),
        G1BeyondMimicPolicyCfg(        # 6: Violin
            policy_name="Violin",
            without_state_estimator=False,
            use_modelmeta_config=True,
            use_motion_from_model=True,
            max_timestep=-1,
            auto_motion_length=True,
            play_speed=1.0,
        ),
    ]

    do_safety_check: bool = True
