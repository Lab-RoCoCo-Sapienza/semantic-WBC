#!/usr/bin/env bash
# Download binary assets from GitHub Releases into this standalone tree.
set -euo pipefail

REL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Standalone: release/ IS the project root (robojudo/ lives here).
# Legacy: release/ nested inside full repo (robojudo/ in parent).
if [[ -d "$REL/robojudo" ]]; then
  ROOT="${1:-$REL}"
elif [[ -d "$REL/../robojudo" ]]; then
  ROOT="${1:-$(cd "$REL/.." && pwd)}"
else
  echo "ERROR: robojudo/ not found in $REL or parent." >&2
  exit 1
fi

VERSION="${ROBOJUDO_ASSETS_VERSION:-1.5.0}"
BASE_URL="${ROBOJUDO_ASSETS_BASE_URL:-https://github.com/michelebri/RoboJuDo/releases/download/v${VERSION}}"
ZIP_NAME="${ROBOJUDO_ASSETS_ZIP:-robojudo-assets-v${VERSION}.zip}"
URL="${ROBOJUDO_ASSETS_URL:-${BASE_URL}/${ZIP_NAME}}"

need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "ERROR: need '$1'" >&2; exit 1; }; }
need_cmd curl
need_cmd unzip

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

echo "Downloading: $URL"
echo "Into:        $ROOT"
curl -fsSL -o "$tmpdir/$ZIP_NAME" "$URL"
unzip -q "$tmpdir/$ZIP_NAME" -d "$tmpdir/extract"

src="$tmpdir/extract"
[[ -d "$tmpdir/extract/release/assets" ]] && src="$tmpdir/extract/release"
[[ -d "$tmpdir/extract/assets" ]] || { echo "ERROR: bad zip layout" >&2; exit 1; }

for dir in assets mp3_songs shazam; do
  [[ -d "$src/$dir" ]] || continue
  mkdir -p "$ROOT/$dir"
  rsync -a "$src/$dir/" "$ROOT/$dir/"
  echo "  ok $dir/"
done

if [[ -f "$src/demo/real_try_trimmed_minus2s.mp3" ]]; then
  cp -f "$src/demo/real_try_trimmed_minus2s.mp3" "$ROOT/demo/"
  cp -f "$src/demo/real_try_trimmed_minus2s.mp3" "$ROOT/" 2>/dev/null || true
  echo "  ok demo/real_try_trimmed_minus2s.mp3"
fi

echo "Done. Test: python scripts/run_pipeline.py -c g1_switch_beyondmimic"
