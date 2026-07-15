import logging

import numpy as np
import onnxruntime as ort

from robojudo.environment.utils.mujoco_viz import MujocoVisualizer
from robojudo.policy import Policy, policy_registry
from robojudo.policy.policy_cfgs import BeyondMimicPolicyCfg
from robojudo.tools.dof import DoFConfig
from robojudo.utils.progress import ProgressBar
from robojudo.utils.rotation import TransformAlignment
from robojudo.utils.util_func import matrix_from_quat, subtract_frame_transforms

logger = logging.getLogger(__name__)


@policy_registry.register
class BeyondMimicPolicy(Policy):
    cfg_policy: BeyondMimicPolicyCfg

    def __init__(self, cfg_policy: BeyondMimicPolicyCfg, device):
        # init onnx, override dof cfg if needed
        sess_options = ort.SessionOptions()

        device = "cpu"
        if device == "cpu":
            providers = ["CPUExecutionProvider"]
        elif device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "tensorrt":
            # Jetson
            providers = [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
        else:
            raise ValueError(f"Unknown device: {device}")

        self.session = ort.InferenceSession(cfg_policy.policy_file, sess_options, providers=providers)

        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.motion_anchor_body_index = -1

        cfg_policy_new = cfg_policy.model_copy()
        if cfg_policy_new.use_modelmeta_config:
            logger.info("[BeyondMimicPolicy] Using modelmeta as config ...")
            modelmeta = self.session.get_modelmeta()  # all str,
            modelmeta_dict = modelmeta.custom_metadata_map

            # dict_keys(['joint_names', 'run_path', 'command_names', 'joint_stiffness', 'joint_damping',
            # 'default_joint_pos', 'action_scale', 'observation_names', 'anchor_body_name', 'body_names'])
            def parse_floats(s):
                return [float(item) for item in s.split(",")]

            def parse_strings(s):
                return [item for item in s.split(",")]

            dof_config = DoFConfig(
                joint_names=parse_strings(modelmeta_dict["joint_names"]),
                default_pos=parse_floats(modelmeta_dict["default_joint_pos"]),
                stiffness=parse_floats(modelmeta_dict["joint_stiffness"]),
                damping=parse_floats(modelmeta_dict["joint_damping"]),
            )
            action_scales = parse_floats(modelmeta_dict["action_scale"])

            anchor_body_name = modelmeta_dict["anchor_body_name"]
            body_names = parse_strings(modelmeta_dict["body_names"])
            self.motion_anchor_body_index = body_names.index(anchor_body_name)

            # command_names = parse_strings(modelmeta_dict["command_names"])
            # observation_names = parse_strings(modelmeta_dict["observation_names"])

            cfg_policy_new.action_dof = dof_config
            cfg_policy_new.obs_dof = dof_config

            cfg_policy_new.action_scales = action_scales

        super().__init__(cfg_policy=cfg_policy_new, device=device)
        self.action_scales = np.asarray(self.cfg_policy.action_scales)
        self.without_state_estimator = self.cfg_policy.without_state_estimator
        self.override_robot_anchor_pos = self.cfg_policy.override_robot_anchor_pos
        self.use_motion_from_model = self.cfg_policy.use_motion_from_model

        self.max_timestep = self.cfg_policy.max_timestep
        self.auto_motion_length = self.cfg_policy.auto_motion_length
        self.motion_length_stable_steps = self.cfg_policy.motion_length_stable_steps
        self.motion_length_pos_eps = self.cfg_policy.motion_length_pos_eps
        self.motion_length_min_active_steps = self.cfg_policy.motion_length_min_active_steps
        # `play_speed` is the per-env-step timeline advance for the motion (timestep += play_speed).
        # Decoupled from `segment_ratio` (which is the wall-clock duration multiplier of the segment
        # time-box). Default 1.0 = motion plays at its native speed; tune per-policy to match BPM.
        speed = float(getattr(self.cfg_policy, "play_speed", 1.0) or 1.0)
        self.play_speed_default = max(1e-3, speed)

        self._auto_ref_joint_pos_prev: np.ndarray | None = None
        self._auto_stable_count: int = 0
        self._auto_active_steps: int = 0
        self._auto_seen_motion: bool = False
        self._auto_motion_start_timestep: float | None = None
        self._auto_estimated_end_timestep: float | None = None

        self.command = None

        action_names = list(self.cfg_action_dof.joint_names)
        locked = list(dict.fromkeys(n for n in self.cfg_policy.locked_joint_names if n))
        self._locked_action_idx = np.array(
            [action_names.index(n) for n in locked if n in action_names],
            dtype=np.int64,
        )
        missing = sorted(set(locked) - set(action_names))
        if missing:
            logger.warning("[BeyondMimicPolicy] locked_joint_names not in action dof (ignored): %s", missing)

        self.reset()

        if self.use_motion_from_model:
            assert self.motion_anchor_body_index >= 0, "motion_anchor_body_index not set"
            assert self.command is not None, "command not initialized"
            command_init = self.command.copy()

            # motion init2anchor alignment
            anchor_pos_w_init = command_init["body_pos_w"][self.motion_anchor_body_index, :]
            anchor_quat_w_init = command_init["body_quat_w"][self.motion_anchor_body_index, :][[1, 2, 3, 0]]

            self.command_init_align = TransformAlignment(
                quat=anchor_quat_w_init, pos=anchor_pos_w_init, yaw_only=True, xy_only=True
            )

    def _prepare_policy(self):
        obs_shape = self.session.get_inputs()[0].shape  # e.g. [1, 154]
        obs = np.zeros(obs_shape[1], dtype=np.float32)
        self.get_action(obs)

    def reset(self):
        self.timestep: float = self.cfg_policy.start_timestep
        if self.use_motion_from_model and self.max_timestep > 0:
            self.pbar = ProgressBar(f"Beyondmimic {self.cfg_policy.policy_name}", self.max_timestep)
        else:
            self.pbar = None
        self.play_speed: float = self.play_speed_default
        self.flag_motion_done = False
        self._auto_ref_joint_pos_prev = None
        self._auto_stable_count = 0
        self._auto_active_steps = 0
        self._auto_seen_motion = False
        self._auto_motion_start_timestep = None
        self._auto_estimated_end_timestep = None
        self._prepare_policy()

    def motion_progress_for_gui(self) -> dict:
        """
        Lightweight progress hint for UI overlays.

        - If `max_timestep > 0`: exact remaining steps until manual cutoff.
        - If `use_motion_from_model` and `max_timestep <= 0`: **HUD-only** progress = elapsed steps /
          `gui_fake_progress_horizon_steps` (does not imply true motion duration from ONNX).
        """
        freq = float(getattr(self.cfg_policy, "freq", 50) or 50)
        name = str(getattr(self.cfg_policy, "policy_name", "BeyondMimic"))

        if self.flag_motion_done:
            return {
                "policy": f"BeyondMimic:{name}",
                "mode": "done",
                "fraction": 1.0,
                "eta_s": 0.0,
                "detail": "motion done (switching...)",
            }

        if self.max_timestep > 0:
            total = float(self.max_timestep)
            t = float(self.timestep)
            rem_steps = max(0.0, total - t)
            frac = 0.0 if total <= 0 else float(np.clip(t / total, 0.0, 1.0))
            return {
                "policy": f"BeyondMimic:{name}",
                "mode": "max_timestep",
                "fraction": frac,
                "eta_s": rem_steps / freq,
                "detail": f"t={t:.1f}/{total:.0f} steps",
            }

        if self.use_motion_from_model and self.max_timestep <= 0:
            # HUD-only timeline: ONNX exports usually don't include a true duration.
            # Bar is purely proportional to elapsed policy steps vs a configurable horizon.
            H = int(getattr(self.cfg_policy, "gui_fake_progress_horizon_steps", 0) or 0)
            t = float(self.timestep)
            t_start = float(self.cfg_policy.start_timestep)
            elapsed = max(0.0, t - t_start)

            if H <= 0:
                return {
                    "policy": f"BeyondMimic:{name}",
                    "mode": "hud_timeline_disabled",
                    "fraction": None,
                    "eta_s": None,
                    "detail": "set gui_fake_progress_horizon_steps > 0 for HUD bar",
                }

            frac = float(np.clip(elapsed / float(H), 0.0, 1.0))
            rem_steps = max(0.0, float(H) - elapsed)
            eta_s = rem_steps / freq

            return {
                "policy": f"BeyondMimic:{name}",
                "mode": "hud_timeline",
                "fraction": frac,
                "eta_s": eta_s,
                "detail": f"HUD {elapsed:.0f}/{H} steps (display only)",
            }

        return {
            "policy": f"BeyondMimic:{name}",
            "mode": "unknown",
            "fraction": None,
            "eta_s": None,
            "detail": "no duration hint (set max_timestep>0 or auto_motion_length=True)",
        }

    def post_step_callback(self, commands: list[str] | None = None):
        self.timestep += 1 * self.play_speed
        if self.pbar:
            self.pbar.set(self.timestep)

        if 0 < self.max_timestep <= self.timestep:
            self.play_speed = 0.0
            self.flag_motion_done = True

        if (
            self.auto_motion_length
            and self.use_motion_from_model
            and self.max_timestep <= 0
            and not self.flag_motion_done
            and self.play_speed > 0
            and self.command is not None
        ):
            ref_pos = np.asarray(self.command["joint_pos"], dtype=np.float64).reshape(-1)
            if self._auto_ref_joint_pos_prev is None:
                self._auto_ref_joint_pos_prev = ref_pos.copy()
            else:
                delta = float(np.max(np.abs(ref_pos - self._auto_ref_joint_pos_prev)))
                if delta > self.motion_length_pos_eps:
                    self._auto_active_steps += 1
                    self._auto_stable_count = 0
                    if self._auto_active_steps >= self.motion_length_min_active_steps:
                        if not self._auto_seen_motion:
                            self._auto_motion_start_timestep = float(self.timestep)
                        self._auto_seen_motion = True
                else:
                    self._auto_stable_count += 1

                self._auto_ref_joint_pos_prev = ref_pos.copy()

                if self._auto_seen_motion and 0 < self._auto_stable_count < self.motion_length_stable_steps:
                    est_end = float(self.timestep) + float(self.motion_length_stable_steps - self._auto_stable_count)
                    if self._auto_estimated_end_timestep is None:
                        self._auto_estimated_end_timestep = est_end
                    else:
                        self._auto_estimated_end_timestep = max(self._auto_estimated_end_timestep, est_end)
                if self._auto_seen_motion and self._auto_stable_count >= self.motion_length_stable_steps:
                    if self._auto_estimated_end_timestep is None:
                        self._auto_estimated_end_timestep = float(self.timestep)
                    else:
                        self._auto_estimated_end_timestep = max(self._auto_estimated_end_timestep, float(self.timestep))

                if self._auto_seen_motion and self._auto_stable_count >= self.motion_length_stable_steps:
                    self.play_speed = 0.0
                    self.flag_motion_done = True
                    logger.info(
                        "[BeyondMimicPolicy] Auto motion end detected "
                        f"({self.cfg_policy.policy_name}: stable_ref_steps={self.motion_length_stable_steps}, "
                        f"eps={self.motion_length_pos_eps})."
                    )

        for command in commands or []:
            match command:
                case "[MOTION_RESET]":
                    self.reset()
                case "[MOTION_FADE_IN]":
                    self.play_speed = self.play_speed_default
                case "[MOTION_FADE_OUT]":
                    self.play_speed = 0.0

    def _get_command(self, env_data, ctrl_data):
        if not self.use_motion_from_model:
            assert "BeyondMimicCtrl" in ctrl_data, "BeyondMimicCtrl not found in ctrl_data"
            command = ctrl_data.get("BeyondMimicCtrl")
            self.command = command
            # print(command.time_steps[0])
            return (
                command.command,
                command.robot_anchor_pos_w,
                command.robot_anchor_quat_w,
                command.anchor_pos_w,
                command.anchor_quat_w,
                command.get("hand_pose", None),
            )
        else:
            assert self.command is not None, "command not initialized"
            # print(self.command["time_step"])
            command = np.concatenate([self.command["joint_pos"], self.command["joint_vel"]], axis=-1)

            anchor_pos_w = self.command["body_pos_w"][self.motion_anchor_body_index, :]
            anchor_quat_w = self.command["body_quat_w"][self.motion_anchor_body_index, :][[1, 2, 3, 0]]

            if self.command_init_align is not None:
                anchor_quat_w, anchor_pos_w = self.command_init_align.align_transform(anchor_quat_w, anchor_pos_w)

            if self.override_robot_anchor_pos:  # OVERRIDE
                robot_anchor_pos_w = anchor_pos_w.copy()
            else:
                base_pos = env_data.torso_pos
                robot_anchor_pos_w = base_pos

            robot_anchor_quat_w = env_data.torso_quat

            return command, robot_anchor_pos_w, robot_anchor_quat_w, anchor_pos_w, anchor_quat_w, None

    def get_observation(self, env_data, ctrl_data):
        dof_pos = env_data.dof_pos
        dof_vel = env_data.dof_vel
        ang_vel = env_data.base_ang_vel
        lin_vel = env_data.base_lin_vel

        command, robot_anchor_pos_w, robot_anchor_quat_w, anchor_pos_w, anchor_quat_w, hand_pose = self._get_command(
            env_data, ctrl_data
        )

        if self.use_motion_from_model and self._locked_action_idx.size and isinstance(command, np.ndarray):
            command = command.copy()
            nq = int(self.num_actions)
            command[self._locked_action_idx] = self.default_dof_pos[self._locked_action_idx]
            command[self._locked_action_idx + nq] = 0.0

        pos, ori = subtract_frame_transforms(
            robot_anchor_pos_w,
            robot_anchor_quat_w,
            anchor_pos_w,
            anchor_quat_w,
        )
        mat = matrix_from_quat(ori)

        obs_command = command
        obs_motion_anchor_pos_b = pos
        obs_motion_anchor_ori_b = mat[:, :2].flatten()

        obs_base_lin_vel = lin_vel
        obs_base_ang_vel = ang_vel
        obs_joint_pos_rel = dof_pos - self.default_dof_pos
        obs_joint_vel_rel = dof_vel
        obs_last_action = self.last_action

        obs_prop = np.concatenate(
            [
                obs_command,
                obs_motion_anchor_pos_b if not self.without_state_estimator else [],
                obs_motion_anchor_ori_b,
                obs_base_lin_vel if not self.without_state_estimator else [],
                obs_base_ang_vel,
                obs_joint_pos_rel,
                obs_joint_vel_rel,
                obs_last_action,
            ]
        )

        obs = obs_prop
        extras = {
            "pos": pos,
            "ori": ori,
            "robot_anchor_pos_w": robot_anchor_pos_w,
            "robot_anchor_quat_w": robot_anchor_quat_w,
            "anchor_pos_w": anchor_pos_w,
            "anchor_quat_w": anchor_quat_w,
            "command": command,
            "hand_pose": hand_pose,
            "CALLBACK": ["[MOTION_DONE]"] if self.flag_motion_done else [],
        }
        return obs, extras

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        ort_inputs = {
            "obs": np.expand_dims(obs, axis=0).astype(np.float32),
            "time_step": np.expand_dims(np.array([int(self.timestep)]), axis=0).astype(np.float32),
        }

        ort_outputs = self.session.run(
            [
                "actions",
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
            ],
            ort_inputs,
        )
        actions: np.ndarray = np.asarray(ort_outputs[0]).squeeze()

        actions = (1 - self.action_beta) * self.last_action + self.action_beta * actions
        if self._locked_action_idx.size:
            actions[self._locked_action_idx] = 0.0
        self.last_action = actions.copy()

        scaled_actions = actions * self.action_scales

        if self.use_motion_from_model:
            jp = np.asarray(ort_outputs[1], dtype=np.float64).squeeze()
            jv = np.asarray(ort_outputs[2], dtype=np.float64).squeeze()
            if self._locked_action_idx.size:
                jp = jp.copy()
                jv = jv.copy()
                jp[self._locked_action_idx] = self.default_dof_pos[self._locked_action_idx]
                jv[self._locked_action_idx] = 0.0
            self.command = {
                "time_step": self.timestep,
                "joint_pos": jp,
                "joint_vel": jv,
                "body_pos_w": np.asarray(ort_outputs[3]).squeeze(),
                "body_quat_w": np.asarray(ort_outputs[4]).squeeze(),  # as [w, x, y, z]
            }
        return scaled_actions

    def get_init_dof_pos(self) -> np.ndarray:
        """
        Return first frame of the reference motion.
        """
        if self.command is not None:
            joint_pos = self.command["joint_pos"].copy()
            if self._locked_action_idx.size:
                joint_pos[self._locked_action_idx] = self.default_dof_pos[self._locked_action_idx]
            return joint_pos
        else:
            return self.default_dof_pos.copy()

    def debug_viz(self, visualizer: MujocoVisualizer, env_data, ctrl_data, extras):
        robot_anchor_pos_w = extras["robot_anchor_pos_w"]
        robot_anchor_quat_w = extras["robot_anchor_quat_w"]
        anchor_pos_w = extras["anchor_pos_w"]
        anchor_quat_w = extras["anchor_quat_w"]

        pos = extras["pos"]
        # ori = extras["ori"]

        visualizer.draw_arrow(anchor_pos_w, anchor_quat_w, [0.2, 0, 0], color=[1, 0, 0, 1], scale=2, id=0)
        visualizer.draw_arrow(
            robot_anchor_pos_w,
            robot_anchor_quat_w,
            [0.2, 0, 0],
            color=[0, 1, 0, 1],
            scale=2,
            id=1,
        )
        visualizer.draw_arrow(robot_anchor_pos_w, robot_anchor_quat_w, pos, color=[0, 1, 1, 1], scale=2, id=2)

        torso_pos = env_data["torso_pos"]
        torso_quat = env_data["torso_quat"]

        visualizer.draw_arrow(torso_pos, torso_quat, [0.2, 0, 0], color=[1, 1, 0, 1], scale=2, id=3)
