from robojudo.config import ASSETS_DIR, Config


class CtrlCfg(Config):
    ctrl_type: str  # name of the controller class

    triggers: dict[str, str] = {}  # trigger conditions
    triggers_extra: dict[str, str] = {}  # extra trigger conditions


class KeyboardCtrlCfg(CtrlCfg):
    ctrl_type: str = "KeyboardCtrl"

    combination_init_buttons: list[str] = ["Key.ctrl_l"]
    """first button in combination, need to be held down to trigger other commands;"""

    triggers: dict[str, str] = {
        "Key.esc": "[SHUTDOWN]",
        # "Key.tab": "[POLICY_TOGGLE]",
        "r": "[SIM_REBORN]",
        "<": "[MOTION_FADE_IN]",  # note: with shift
        ">": "[MOTION_FADE_OUT]",  # note: with shift
        "|": "[MOTION_RESET]",  # note: with shift
        "{": "[MOTION_LOAD_PREV]",  # note: with shift
        "}": "[MOTION_LOAD_NEXT]",  # note: with shift
    }


class JoystickCtrlCfg(CtrlCfg):
    ctrl_type: str = "JoystickCtrl"

    combination_init_buttons: list[str] = ["LB", "RB"]
    """first button in combination, need to be held down to trigger other commands;"""

    # reference for button names in JoystickThread config
    triggers: dict[str, str] = {
        "A": "[SHUTDOWN]",
        "X": "[MOTION_FADE_IN]",
        "B": "[MOTION_FADE_OUT]",
        "Y": "[MOTION_RESET]",
        # "LB": "[MOTION_LOAD_PREV]",
        # "RB": "[MOTION_LOAD_NEXT]",
        # Note: combo keys supported: "LB+RB+A": "[TEST]",
    }


class UnitreeCtrlCfg(JoystickCtrlCfg):
    ctrl_type: str = "UnitreeCtrl"

    combination_init_buttons: list[str] = ["L1", "R1"]
    """first button in combination, need to be held down to trigger other commands;"""

    triggers: dict[str, str] = {
        "A": "[SHUTDOWN]",
        "X": "[MOTION_FADE_IN]",
        "B": "[MOTION_FADE_OUT]",
        "Y": "[MOTION_RESET]",
        # Note: combo keys supported: "L1+R1+A": "[TEST]",
    }


class MotionCtrlCfg(CtrlCfg):
    class PhcCfg(Config):
        robot_config_file: str
        robot_config: dict = {}  # PLACEHOLDER for phc robot config, to be parsed by config manager

        def model_post_init(self, context) -> None:
            import yaml

            from robojudo.config import THIRD_PARTY_DIR

            # parse phc configs
            phc_dir_path = THIRD_PARTY_DIR / "phc"
            phc_robot_config_file = self.robot_config_file
            phc_robot_config_file_path = phc_dir_path / "phc/data/cfg" / phc_robot_config_file
            if phc_robot_config_file_path.exists():
                phc_robot_config_dict = yaml.safe_load(phc_robot_config_file_path.open("r"))
                phc_robot_config_dict["asset"]["assetRoot"] = phc_dir_path.as_posix()
                phc_robot_config_dict["asset"]["assetFileName"] = (
                    phc_dir_path / phc_robot_config_dict["asset"]["assetFileName"]
                ).as_posix()
                # phc_robot_config_dict["asset"]["urdfFileName"] = (
                #     phc_dir_path / phc_robot_config_dict["asset"]["urdfFileName"]
                # ).as_posix()

                self.robot_config = phc_robot_config_dict

    ctrl_type: str = "MotionCtrl"

    motion_ctrl_gui: bool = True

    # ==== policy specific configs ====
    track_keypoints_names: list[str] = []
    phc: PhcCfg

    # ==== motion config ====
    robot: str
    motion_name: str = ""

    @property
    def motion_path(self) -> str:
        motion_path = ASSETS_DIR / f"motions/{self.robot}/phc/{self.motion_name}.pkl"
        return motion_path.as_posix()


class MotionH2HCtrlCfg(MotionCtrlCfg):
    ctrl_type: str = "MotionH2HCtrl"

    extra_motion_data: bool = False  # extra data for motion recognition


class MotionKungfuBotCtrlCfg(MotionCtrlCfg):
    ctrl_type: str = "MotionKungfuBotCtrl"

    future_max_steps: int = 95
    future_num_steps: int = 20

    anchor_index: int = 0  # root
    key_body_id: list[int]


class MotionTwistCtrlCfg(MotionCtrlCfg):
    ctrl_type: str = "MotionTwistCtrl"

    # ==== motion config ====
    robot: str


class BeyondMimicCtrlCfg(CtrlCfg):
    ctrl_type: str = "BeyondMimicCtrl"

    override_robot_anchor_pos: bool = False  # if True, drop pos fdb

    # ==== motion config ====
    robot: str
    motion_name: str

    @property
    def motion_path(self) -> str:
        motion_path = ASSETS_DIR / f"motions/{self.robot}/beyondmimic/{self.motion_name}.npz"
        return motion_path.as_posix()

    # ==== from beyondmimic ====
    class MotionCommandCfg(Config):
        """Configuration for the motion command."""

        anchor_body_name: str
        body_names: list[str]
        body_names_all: list[str]
        """from beyondmimic asset, used for indexing"""

    motion_cfg: MotionCommandCfg


class TwistRedisCtrlCfg(CtrlCfg):
    ctrl_type: str = "TwistRedisCtrl"

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_key: str = "action_mimic_g1"  # key to get command data from redis

    buffer_size: int = 5  # size of the data buffer to store recent commands


class ShazamFileWatchCtrlCfg(CtrlCfg):
    """Watch a directory for streaming audio chunks, identify them via the local
    Shazam matcher, and queue a matching policy id for the multi-policy pipeline.
    """

    ctrl_type: str = "ShazamFileWatchCtrl"

    watch_dir: str = "runtime_chunks"
    """Directory where rolling audio chunks (NNNN.mp3/.wav/.flac/.m4a) are dropped."""

    clean_watch_dir_on_start: bool = True
    """If True, remove existing chunk files in ``watch_dir`` when the controller starts
    (same extensions / ``watch_name_pattern`` rules as the watcher). Set False to keep
    old clips across pipeline restarts."""

    index_path: str = "shazam/index.pkl"
    """Pickle index produced by `shazam/run_experiment.py --build-index`."""

    clap_index_path: str | None = "shazam/clap_index.json"
    """Optional CLAP index JSON. Set to None to disable CLAP fallback."""

    clap_fallback: bool = True
    """If True, rank by CLAP embedding similarity when fingerprint has no/weak match."""

    clap_fallback_if_fp_votes_below: int = 50
    """Trigger CLAP fallback when the fingerprint votes are below this threshold."""

    clap_min_sim: float = 0.35
    clap_embed_seconds: float = 10.0
    clap_model: str = "laion/larger_clap_general"
    clap_device: str = "cpu"

    song_to_policy: dict[str, int] = {}
    """Case-insensitive substring -> policy id. First matching key wins.
    Examples: {"salsa": 7, "dynamite": 3, "swim": 5}."""

    song_time_to_policy: list[tuple[str, float, float, int]] = []
    """Optional per-song time-window mapping, evaluated *before* ``song_to_policy``.

    Each entry is ``(song_substring, start_s, end_s, policy_id)`` where the range
    is half-open ``[start_s, end_s)``. Use ``end_s <= 0`` (e.g. ``-1.0``) as the
    open upper bound (``start_s..inf``). The first entry whose key matches the
    song and whose ``[start_s, end_s)`` contains the Shazam ``offset_sec`` wins.

    If the matcher returns no offset, or no range matches, the controller falls
    back to ``song_to_policy``.

    Example (same song, different policies per section)::

        song_time_to_policy=[
            ("thriller", 0.0, 30.0, 4),
            ("thriller", 30.0, 60.0, 7),
            ("thriller", 60.0, -1.0, 9),
            ("salsa", 0.0, 40.0, 2),
            ("salsa", 40.0, -1.0, 6),
        ]
    """

    min_confidence: float = 0.0
    """Drop matches below this confidence (set to 0 to accept everything)."""

    poll_interval_s: float = 0.25
    """How often (seconds) the background thread scans the watch_dir."""

    delete_after: bool = False
    """If True, delete chunk files after they are identified."""

    queue_maxlen: int = 4
    """Max number of pending policy ids kept in the queue (oldest dropped)."""

    supported_exts: list[str] = [".mp3", ".wav", ".flac", ".m4a"]

    watch_name_pattern: str | None = None
    """Regex matched against each filename (not full path). If set, files that do not
    match are ignored — use e.g. ``r'^\\d{4}\\.'`` for ``0007.mp3`` streams and skip
    unrelated names like ``BTS_-_Butter_Lyrics_0007.mp3``."""

    drop_stale_chunks: bool = True
    """If True, the pipeline drops queue heads whose ``chunk_index`` is older
    than the last applied hit (to avoid re-playing stale segments)."""

    max_chunk_age_s: float = 12.0
    """Max age (monotonic seconds) of a queued hit before it is considered
    stale and dropped at the next segment boundary. Set to <=0 to disable."""

    controller_delay_s: float = 0.0
    """Hold each Shazam hit inside the controller for this many seconds before
    exposing it to the pipeline. Used to compensate audio playback latency so
    the policy switches when the user actually *hears* the chunk, not when the
    streamer drops the file. ``0.0`` disables the delay."""


class MusicCommandSocketCtrlCfg(CtrlCfg):
    """TCP command listener for external music->policy bridges.

    Expected message format is either:
    - plain text command line, e.g. ``[POLICY_SWITCH],5``
    - JSON line with ``{"command": "[POLICY_SWITCH],5"}``
    """

    ctrl_type: str = "MusicCommandSocketCtrl"
    host: str = "0.0.0.0"
    port: int = 8765
    queue_maxlen: int = 32


class MusicCommandSocketClientCtrlCfg(CtrlCfg):
    """TCP client for receiving remote command strings from an external server."""

    ctrl_type: str = "MusicCommandSocketClientCtrl"
    host: str = "127.0.0.1"
    port: int = 8765
    queue_maxlen: int = 32
    subscribe_topic: str = "gesture"
    connect_timeout_s: float = 2.0
    reconnect_interval_s: float = 1.0


class PolicyGuiCtrlCfg(CtrlCfg):
    ctrl_type: str = "PolicyGuiCtrl"
