# Fix OMP perfmance issue on ARM platform (Jetson)
import os
import platform

if platform.machine().startswith("aarch64"):
    os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import logging
import time

import robojudo.pipeline
from robojudo.config.config_manager import ConfigManager
from robojudo.controller.ctrl_cfgs import MusicCommandSocketClientCtrlCfg
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg
from robojudo.pipeline.rl_pipeline import RlPipeline

logger = logging.getLogger("robojudo")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run RoboJuDo pipeline with optional remote TCP command client (robot side).",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="g1",
        help="Name of the config class to use",
    )
    parser.add_argument(
        "--cmd-server-host",
        type=str,
        default="",
        help="Connect to this PC/listener host for remote commands (TCP client mode).",
    )
    parser.add_argument(
        "--cmd-server-port",
        type=int,
        default=8765,
        help="Remote command server port (default: 8765).",
    )
    parser.add_argument(
        "--cmd-subscribe-topic",
        type=str,
        default="gesture",
        help="SUBSCRIBE topic sent on connect (default: gesture).",
    )
    parser.add_argument(
        "--cmd-duration-unit",
        type=str,
        default="seconds",
        choices=("seconds",),
        help="Reserved for listener duration semantics (currently informational).",
    )
    parser.add_argument(
        "--cmd-debug-only",
        action="store_true",
        help="Only start the TCP client and log received lines (no robot pipeline).",
    )
    return parser.parse_args()


def _inject_remote_client_ctrl(cfg: RlPipelineCfg, args: argparse.Namespace) -> None:
    host = str(args.cmd_server_host or "").strip()
    if not host:
        return
    client_cfg = MusicCommandSocketClientCtrlCfg(
        host=host,
        port=int(args.cmd_server_port),
        subscribe_topic=str(args.cmd_subscribe_topic or "gesture").strip() or "gesture",
        queue_maxlen=32,
        connect_timeout_s=10.0,
        reconnect_interval_s=0.5,
    )
    cfg.ctrl = list(cfg.ctrl) + [client_cfg]
    logger.info(
        "Remote command client enabled -> %s:%d topic=%s unit=%s",
        host,
        int(args.cmd_server_port),
        client_cfg.subscribe_topic,
        args.cmd_duration_unit,
    )


def main():
    args = parse_args()
    logger.info(f"Using config: {args.config}")
    config_manager = ConfigManager(config_name=args.config)
    cfg: RlPipelineCfg = config_manager.get_cfg()

    _inject_remote_client_ctrl(cfg, args)

    if args.cmd_debug_only:
        if not str(args.cmd_server_host or "").strip():
            raise SystemExit("--cmd-debug-only requires --cmd-server-host")
        from robojudo.controller.music_command_socket_client_ctrl import MusicCommandSocketClientCtrl

        client_cfg = MusicCommandSocketClientCtrlCfg(
            host=str(args.cmd_server_host).strip(),
            port=int(args.cmd_server_port),
            subscribe_topic=str(args.cmd_subscribe_topic or "gesture").strip() or "gesture",
            queue_maxlen=32,
            connect_timeout_s=10.0,
            reconnect_interval_s=0.5,
        )
        ctrl = MusicCommandSocketClientCtrl(cfg_ctrl=client_cfg)
        logger.info("cmd-debug-only: waiting for remote lines (Ctrl+C to stop)")
        try:
            while True:
                _, commands = ctrl.process_triggers({})
                for line in commands:
                    logger.info("[cmd-debug-only] received: %s", line)
                time.sleep(0.05)
        except KeyboardInterrupt:
            logger.info("cmd-debug-only: stopped")
        finally:
            ctrl.shutdown()
        return

    pipeline_type = cfg.pipeline_type
    pipeline_class: type[RlPipeline] = getattr(robojudo.pipeline, pipeline_type)
    logger.info(f"Using pipeline: {pipeline_type} -> {pipeline_class}")

    pipeline = pipeline_class(cfg=cfg)

    if not cfg.env.is_sim:
        pipeline.prepare()

    while True:
        time_start = time.time()
        pipeline.step()
        time_end = time.time()
        time_diff = time_end - time_start

        if not cfg.run_fullspeed:
            time_diff = pipeline.dt - time_diff
            if time_diff > 0:
                time.sleep(time_diff)
            else:
                if not cfg.env.is_sim:
                    logger.error(f"Warning: frame drop -> {time_diff}")
                    if time_diff < -0.2:
                        logger.critical("Exiting due to excessive frame drop")
                        pipeline.env.shutdown()
                        time.sleep(10)
                        break


if __name__ == "__main__":
    main()
