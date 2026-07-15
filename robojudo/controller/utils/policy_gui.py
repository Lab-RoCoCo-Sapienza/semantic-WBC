import tkinter as tk
from threading import Thread


class PolicyGUI:
    def __init__(self, policy_ctrl):
        self.policy_ctrl = policy_ctrl

        self.thread_gui = Thread(target=self._run_gui, daemon=True)
        self.thread_gui.start()

    def _emit(self, command: str):
        self.policy_ctrl.gui_commands.put(command)

    def _run_gui(self):
        root = tk.Tk()
        root.title("RoboJuDo Policy Panel")
        root.attributes("-topmost", True)

        frame = tk.Frame(root, padx=10, pady=10)
        frame.pack(fill="both", expand=True)

        row = 0

        tk.Label(frame, text="Unitree modes", font=("Arial", 12, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(0, 6)
        )
        row += 1

        tk.Button(frame, text="W: Walk", width=18, command=lambda: self._emit("[POLICY_SWITCH],0")).grid(
            row=row, column=0, padx=(0, 8), pady=4, sticky="ew"
        )
        tk.Button(frame, text="S: Stand", width=18, command=lambda: self._emit("[POLICY_SWITCH],1")).grid(
            row=row, column=1, pady=4, sticky="ew"
        )
        row += 1

        tk.Label(frame, text="BeyondMimic models", font=("Arial", 12, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(10, 6)
        )
        row += 1

        buttons = [
            ("1: bts_2_0_tracking_2", "[POLICY_SWITCH],2"),
            ("2: bts_dynamite_tracking", "[POLICY_SWITCH],3"),
            ("3: easy_sample", "[POLICY_SWITCH],4"),
            ("4: Swim_tracking", "[POLICY_SWITCH],5"),
            ("5: thriller", "[POLICY_SWITCH],6"),
            ("6: salsa_tracking", "[POLICY_SWITCH],7"),
        ]
        for label, cmd in buttons:
            tk.Button(frame, text=label, width=40, command=lambda c=cmd: self._emit(c)).grid(
                row=row, column=0, columnspan=2, pady=3, sticky="ew"
            )
            row += 1

        tk.Label(frame, text="System", font=("Arial", 12, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(10, 6)
        )
        row += 1

        tk.Button(frame, text="R: Reborn", command=lambda: self._emit("[SIM_REBORN]")).grid(
            row=row, column=0, padx=(0, 8), pady=4, sticky="ew"
        )
        tk.Button(frame, text="Esc Shutdown", command=lambda: self._emit("[SHUTDOWN]")).grid(
            row=row, column=1, pady=4, sticky="ew"
        )

        for c in range(2):
            frame.grid_columnconfigure(c, weight=1)

        root.mainloop()

