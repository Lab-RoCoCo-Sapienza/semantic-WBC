from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class _Button:
    label: str
    command: str
    rect: tuple[int, int, int, int]  # x, y, w, h


def run_policy_gui(queue_out, queue_in=None, width: int = 420, height: int = 580):
    """
    Run pygame window in the *main thread of this process*.
    Send clicked commands to queue_out.
    """
    import pygame

    pygame.init()
    pygame.display.set_caption("RoboJuDo Policy Panel")
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("Arial", 16)
    font_h = pygame.font.SysFont("Arial", 18, bold=True)
    font_small = pygame.font.SysFont("Arial", 14)

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

    progress = {
        "title": "Policy progress",
        "policy": "—",
        "detail": "—",
        "fraction": None,  # None => indeterminate
        "eta_s": None,
        "phase": "",
    }

    def draw():
        progress_y = min(max(10, y + 10), height - 120)
        progress_rect = pygame.Rect(10, progress_y, width - 20, 104)

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

        # Progress panel
        pygame.draw.rect(screen, (30, 30, 36), progress_rect, border_radius=8)
        pygame.draw.rect(screen, (90, 90, 110), progress_rect, width=1, border_radius=8)

        title_s = font_h.render(str(progress.get("title", "Policy progress")), True, (220, 220, 220))
        screen.blit(title_s, (progress_rect.x + 10, progress_rect.y + 8))

        pol = str(progress.get("policy", "—"))
        phase = str(progress.get("phase", "")).strip()
        head = pol if not phase else f"{pol}  ({phase})"
        pol_s = font.render(head, True, (230, 230, 230))
        screen.blit(pol_s, (progress_rect.x + 10, progress_rect.y + 34))

        det = str(progress.get("detail", ""))
        det_s = font_small.render(det[:120], True, (180, 180, 190))
        screen.blit(det_s, (progress_rect.x + 10, progress_rect.y + 56))

        bar_x = progress_rect.x + 10
        bar_y = progress_rect.y + 78
        bar_w = progress_rect.w - 20
        bar_h = 16
        pygame.draw.rect(screen, (18, 18, 22), pygame.Rect(bar_x, bar_y, bar_w, bar_h), border_radius=6)
        frac = progress.get("fraction", None)
        if frac is None:
            # indeterminate shimmer
            t = pygame.time.get_ticks() / 1000.0
            w = int(bar_w * 0.25)
            x = bar_x + int((bar_w - w) * (0.5 + 0.5 * math.sin(t * 6.0)))
            pygame.draw.rect(screen, (80, 140, 255), pygame.Rect(x, bar_y, w, bar_h), border_radius=6)
        else:
            try:
                f = float(frac)
            except Exception:
                f = 0.0
            f = max(0.0, min(1.0, f))
            fill_w = int(bar_w * f)
            if fill_w > 0:
                pygame.draw.rect(screen, (70, 200, 120), pygame.Rect(bar_x, bar_y, fill_w, bar_h), border_radius=6)

        eta = progress.get("eta_s", None)
        if eta is None:
            eta_txt = "ETA: —"
        else:
            try:
                eta_txt = f"ETA: {float(eta):.2f}s"
            except Exception:
                eta_txt = "ETA: —"
        eta_s = font_small.render(eta_txt, True, (180, 180, 190))
        screen.blit(eta_s, (bar_x + bar_w - eta_s.get_width(), progress_rect.y + 56))

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
                        try:
                            queue_out.put_nowait(b.command)
                        except Exception:
                            pass
                        break

        if queue_in is not None:
            while True:
                try:
                    msg = queue_in.get_nowait()
                except Exception:
                    break
                if not isinstance(msg, dict):
                    continue
                if msg.get("type") != "progress":
                    continue
                for k in ("policy", "detail", "fraction", "eta_s", "phase", "title"):
                    if k in msg:
                        progress[k] = msg[k]

        draw()
        clock.tick(30)

    pygame.quit()

