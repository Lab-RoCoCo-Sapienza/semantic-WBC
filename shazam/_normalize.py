"""Title normalization shared between run_experiment and compare_report.

Dataset folder names and source song filenames use different separators
(`BTS_-_Dynamite_Lyrics` vs `BTS - Dynamite (Lyrics).mp3`), so we reduce
both to a canonical form before comparing.
"""

import re


def canon_title(value: str | None) -> str:
    if not value:
        return ""
    lowered = value.lower()
    cleaned = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(cleaned.split())
