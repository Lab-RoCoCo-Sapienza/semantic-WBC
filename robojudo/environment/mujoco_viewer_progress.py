from __future__ import annotations

import time
from typing import Any


def _ascii(s: Any, max_len: int = 160) -> str:
    """
    `mujoco.mjr_overlay` text is effectively ASCII-only in practice; unicode glyphs often render as garbage.
    """
    if s is None:
        return ""
    if isinstance(s, bytes):
        try:
            s = s.decode("utf-8", errors="ignore")
        except Exception:
            s = str(s)
    t = str(s)
    t = t.encode("ascii", errors="ignore").decode("ascii", errors="ignore")
    t = " ".join(t.split())
    if len(t) > max_len:
        t = t[: max_len - 3] + "..."
    return t


def _strip_robojudo_bottom_right_legend(viewer: Any) -> None:
    """Remove duplicate policy key-help panel (forked viewer) so one HUD block remains."""
    try:
        mujoco = __import__("mujoco")
        br = mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT
        block = viewer._overlay.get(br)  # type: ignore[attr-defined]
        if not block or len(block) < 2:
            return
        blob = (block[0] or "") + (block[1] or "")
        if "robojudo" in blob.lower():
            del viewer._overlay[br]  # type: ignore[attr-defined]
    except Exception:
        return


def _merge_legend_rows(legend_rows: list[Any]) -> str:
    parts: list[str] = []
    for row in legend_rows:
        if isinstance(row, (list, tuple)) and len(row) >= 2:
            a, b = str(row[0]).strip(), str(row[1]).strip()
            if a and b:
                parts.append(f"{a}: {b}")
            elif b:
                parts.append(b)
            elif a:
                parts.append(a)
        elif isinstance(row, str) and row.strip():
            parts.append(row.strip())
    return " | ".join(parts)


def install_mujoco_viewer_progress_hook(viewer: Any) -> None:
    """
    Inject a compact two-column overlay into `mujoco-python-viewer`'s HUD (TOPRIGHT).

    Progress, ETA, remain text, progress bar, and hotkeys are laid out as a few wide
    rows (pipe-separated on the right). When ``payload["shazam_hud"]`` is set, a second
    block is written to BOTTOMRIGHT (chunk vs motion) without touching TOPRIGHT.

    Duplicate RoboJuDo key panels in BOTTOMRIGHT (common in forks) are removed when
    detected (substring ``robojudo``), then Shazam lines are appended.
    """
    if viewer is None:
        return
    if getattr(viewer, "_robojudo_mujoco_overlay_hook_installed", False):
        return

    orig = viewer._create_overlay  # type: ignore[attr-defined]

    def _wrapped_create_overlay():  # noqa: ANN202
        orig()
        _strip_robojudo_bottom_right_legend(viewer)
        try:
            mujoco = __import__("mujoco")
            topright = mujoco.mjtGridPos.mjGRID_TOPRIGHT

            if topright not in viewer._overlay:  # type: ignore[attr-defined]
                viewer._overlay[topright] = ["", ""]  # type: ignore[attr-defined]

            def add_line(left: str, right: str, right_max: int = 220):
                viewer._overlay[topright][0] += _ascii(left) + "\n"  # type: ignore[attr-defined]
                viewer._overlay[topright][1] += _ascii(right, max_len=right_max) + "\n"  # type: ignore[attr-defined]

            state = getattr(viewer, "_robojudo_progress_state", None)
            if not isinstance(state, dict):
                return

            title = _ascii(state.get("title", "Policy progress"))
            policy = _ascii(state.get("policy", "-"))
            detail = _ascii(state.get("detail", "-"), max_len=120)
            phase = _ascii(state.get("phase", "")).strip()
            if phase:
                policy_line = f"{policy}  ({phase})"
            else:
                policy_line = policy

            frac = state.get("fraction", None)
            eta = state.get("eta_s", None)
            if eta is None:
                eta_txt = "n/a"
            else:
                try:
                    eta_txt = f"{float(eta):.2f}s"
                except Exception:
                    eta_txt = "n/a"

            width = 26
            if frac is None:
                t = time.time()
                shift = int((t * 6.0) % width)
                pat = ["."] * width
                for i in range(max(0, width // 4)):
                    idx = (shift + i) % width
                    pat[idx] = "#"
                bar = "".join(pat)
            else:
                try:
                    f = float(frac)
                except Exception:
                    f = 0.0
                f = max(0.0, min(1.0, f))
                filled = int(round(f * width))
                bar = "#" * filled + "-" * (width - filled) + f" {f * 100:5.1f}%"

            status_right = f"{policy_line} | ETA {eta_txt} | {detail}"
            add_line(title, status_right, right_max=240)
            add_line("Bar", bar, right_max=80)

            legend_rows = state.get("legend_rows")
            if isinstance(legend_rows, list) and legend_rows:
                merged = _merge_legend_rows(legend_rows)
                if merged:
                    add_line("Hotkeys", merged, right_max=260)

            shazam_hud = state.get("shazam_hud")
            if isinstance(shazam_hud, dict) and shazam_hud.get("show"):
                br = mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT
                if br not in viewer._overlay:  # type: ignore[attr-defined]
                    viewer._overlay[br] = ["", ""]  # type: ignore[attr-defined]
                hud_lines = shazam_hud.get("lines")
                if isinstance(hud_lines, list):
                    for row in hud_lines:
                        if isinstance(row, (list, tuple)) and len(row) >= 2:
                            lft, rgt = row[0], row[1]
                        else:
                            continue
                        viewer._overlay[br][0] += _ascii(lft) + "\n"  # type: ignore[attr-defined]
                        viewer._overlay[br][1] += _ascii(rgt, max_len=300) + "\n"  # type: ignore[attr-defined]
        except Exception:
            return

    viewer._create_overlay = _wrapped_create_overlay  # type: ignore[attr-defined]
    viewer._robojudo_mujoco_overlay_hook_installed = True


def set_mujoco_viewer_progress_state(viewer: Any, payload: dict) -> None:
    if viewer is None:
        return
    viewer._robojudo_progress_state = payload  # type: ignore[attr-defined]
