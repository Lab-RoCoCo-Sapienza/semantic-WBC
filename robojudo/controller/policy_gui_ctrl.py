from queue import Empty, Queue

from robojudo.controller import Controller, ctrl_registry
from robojudo.controller.ctrl_cfgs import CtrlCfg


def publish_policy_gui_progress(ctrl_manager, payload: dict) -> None:
    """
    Best-effort UI update for `PolicyGuiCtrl` (pygame panel).

    `payload` should be JSON-serializable-ish (small dict).
    """
    try:
        gui = ctrl_manager.controllers.get("PolicyGuiCtrl", None)
        if gui is None:
            return
        inst = gui.inst
        q = getattr(inst, "_progress_queue", None)
        if q is None:
            return
        msg = {"type": "progress", **(payload or {})}
        try:
            q.put_nowait(msg)
        except Exception:
            # Drop if the UI is lagging; latest state will be sent again next step.
            try:
                _ = q.get_nowait()
            except Exception:
                pass
            try:
                q.put_nowait(msg)
            except Exception:
                pass
    except Exception:
        return


@ctrl_registry.register
class PolicyGuiCtrl(Controller):
    """
    Small Tkinter panel that emits pipeline COMMANDS on button clicks.
    """

    cfg_ctrl: CtrlCfg

    def __init__(self, cfg_ctrl: CtrlCfg, env=None, **kwargs):
        super().__init__(cfg_ctrl=cfg_ctrl, env=env, **kwargs)
        self.gui_commands: Queue[str] = Queue(maxsize=200)
        # Create a real Python window (pygame) in a separate process.
        # On macOS, NSWindow must be created on the main thread of its process.
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        self._ipc_queue = ctx.Queue(maxsize=200)
        self._progress_queue = ctx.Queue(maxsize=5)
        from robojudo.controller.utils.policy_gui_pygame_process import run_policy_gui  # noqa: WPS433

        self._gui_process = ctx.Process(
            target=run_policy_gui,
            args=(self._ipc_queue, self._progress_queue),
            daemon=True,
        )
        self._gui_process.start()

    def reset(self):
        while not self.gui_commands.empty():
            try:
                self.gui_commands.get_nowait()
            except Empty:
                break
        while hasattr(self, "_ipc_queue") and not self._ipc_queue.empty():
            try:
                self._ipc_queue.get_nowait()
            except Exception:
                break
        while hasattr(self, "_progress_queue") and not self._progress_queue.empty():
            try:
                self._progress_queue.get_nowait()
            except Exception:
                break

    def get_data(self):
        # no sensor/control data; only commands from GUI
        return {}

    def process_triggers(self, ctrl_data):
        commands: list[str] = []
        # Pull commands from GUI process.
        if hasattr(self, "_ipc_queue"):
            while not self._ipc_queue.empty():
                try:
                    self.gui_commands.put_nowait(self._ipc_queue.get_nowait())
                except Exception:
                    break
        while not self.gui_commands.empty():
            try:
                commands.append(self.gui_commands.get_nowait())
            except Empty:
                break
        return ctrl_data, commands

