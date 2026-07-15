from __future__ import annotations

from dataclasses import dataclass
from threading import Thread


@dataclass(frozen=True)
class _Button:
    label: str
    command: str
    rect: tuple[int, int, int, int]  # x, y, w, h


class PolicyGUI:
    """
    Small clickable policy panel implemented with pygame.
    Fallback for systems where Tkinter is unavailable.
    """

    def __init__(self, policy_ctrl):
        self.policy_ctrl = policy_ctrl
        self.thread_gui = Thread(target=self._run_gui, daemon=True)
        self.thread_gui.start()

    def _emit(self, command: str):
        self.policy_ctrl.gui_commands.put(command)

    def _run_gui(self):
        import pygame

        pygame.init()
        pygame.display.set_caption("RoboJuDo Policy Panel")

        width, height = 380, 430
        screen = pygame.display.set_mode((width, height))
        clock = pygame.time.Clock()

        font = pygame.font.SysFont("Arial", 16)
        font_h = pygame.font.SysFont("Arial", 18, bold=True)

        buttons: list[_Button] = []

        y = 10
        section_gap = 10
        btn_h = 34
        btn_w = width - 20
        x = 10

        def add_header(text: str):
            nonlocal y
            buttons.append(_Button(label=f"__HEADER__{text}", command="", rect=(0, y, 0, 0)))
            y += 26

        def add_button(label: str, command: str):
            nonlocal y
            buttons.append(_Button(label=label, command=command, rect=(x, y, btn_w, btn_h)))
            y += btn_h + 8

        add_header("Unitree modes")
        add_button("W: Walk", "[POLICY_SWITCH],0")
        add_button("S: Stand", "[POLICY_SWITCH],1")
        y += section_gap

        add_header("BeyondMimic models")
        add_button("1: bts_2_0_tracking_2", "[POLICY_SWITCH],2")
        add_button("2: bts_dynamite_tracking", "[POLICY_SWITCH],3")
        add_button("3: easy_sample", "[POLICY_SWITCH],4")
        add_button("4: Swim_tracking", "[POLICY_SWITCH],5")
        add_button("5: thriller", "[POLICY_SWITCH],6")
        add_button("6: salsa_tracking", "[POLICY_SWITCH],7")
        add_button("7: gdance", "[POLICY_SWITCH],8")
        add_button("8: Salsa_4", "[POLICY_SWITCH],9")
        add_button("9: thriller_locked_waist", "[POLICY_SWITCH],10")
        y += section_gap

        add_header("System")
        add_button("R: Reborn", "[SIM_REBORN]")
        add_button("Esc Shutdown", "[SHUTDOWN]")

        def draw():
            screen.fill((24, 24, 28))
            for b in buttons:
                if b.label.startswith("__HEADER__"):
                    text = b.label.replace("__HEADER__", "")
                    surf = font_h.render(text, True, (220, 220, 220))
                    screen.blit(surf, (10, b.rect[1]))
                    continue

                rx, ry, rw, rh = b.rect
                pygame.draw.rect(screen, (50, 50, 60), pygame.Rect(rx, ry, rw, rh), border_radius=6)
                pygame.draw.rect(screen, (90, 90, 110), pygame.Rect(rx, ry, rw, rh), width=1, border_radius=6)
                surf = font.render(b.label, True, (240, 240, 240))
                screen.blit(surf, (rx + 10, ry + 8))

            pygame.display.flip()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    for b in buttons:
                        if not b.command:
                            continue
                        rx, ry, rw, rh = b.rect
                        if rx <= mx <= rx + rw and ry <= my <= ry + rh:
                            self._emit(b.command)
                            break

            draw()
            clock.tick(30)

        pygame.quit()

