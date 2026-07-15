#!/usr/bin/env bash
# Install runtime assets for semantic-WBC:
#   1) base RoboJuDo assets from upstream HansZ8/RoboJuDo (always)
#   2) optional semantic-WBC extras zip (demo ONNX, mp3, Shazam) when available
set -euo pipefail

REL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -d "$REL/robojudo" ]]; then
  ROOT="${1:-$REL}"
elif [[ -d "$REL/../robojudo" ]]; then
  ROOT="${1:-$(cd "$REL/.." && pwd)}"
else
  echo "ERROR: robojudo/ not found in $REL or parent." >&2
  exit 1
fi

UPSTREAM_REPO="${ROBOJUDO_UPSTREAM_REPO:-https://github.com/HansZ8/RoboJuDo.git}"
UPSTREAM_REF="${ROBOJUDO_UPSTREAM_REF:-release}"

VERSION="${ROBOJUDO_ASSETS_VERSION:-1.5.0}"
EXTRAS_BASE_URL="${ROBOJUDO_ASSETS_BASE_URL:-https://github.com/Lab-RoCoCo-Sapienza/semantic-WBC/releases/download/v${VERSION}}"
EXTRAS_ZIP_NAME="${ROBOJUDO_ASSETS_ZIP:-robojudo-assets-v${VERSION}.zip}"
EXTRAS_URL="${ROBOJUDO_ASSETS_URL:-${EXTRAS_BASE_URL}/${EXTRAS_ZIP_NAME}}"
SKIP_EXTRAS="${ROBOJUDO_SKIP_EXTRAS:-0}"

SEMANTIC_ONNX=(
  bts_2_0_tracking_2.onnx
  bts_dynamite_tracking.onnx
  easy_sample.onnx
  Swim_tracking.onnx
  thriller.onnx
  salsa_tracking.onnx
  gdance.onnx
  Salsa_4.onnx
  thriller_locked_waist.onnx
  g1_29dof_gesti_10k.onnx
  g1_29dof_gesti.onnx
  g1_29dof_67_10k.onnx
  g1_29dof_67_safe.onnx
  g1_29dof_gesti_safe.onnx
)

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: need '$1'" >&2
    exit 1
  }
}

fetch_upstream_assets() {
  local tmpdir="$1"
  local dest="$ROOT/assets"

  echo "==> Fetching base assets from ${UPSTREAM_REPO} (${UPSTREAM_REF})"

  if command -v git >/dev/null 2>&1 && command -v rsync >/dev/null 2>&1; then
    local clone_dir="$tmpdir/upstream"
    git clone --depth 1 --filter=blob:none --sparse -b "$UPSTREAM_REF" "$UPSTREAM_REPO" "$clone_dir"
    (
      cd "$clone_dir"
      git sparse-checkout set assets
    )
    mkdir -p "$dest"
    rsync -a "$clone_dir/assets/" "$dest/"
    echo "  ok assets/ (sparse clone)"
    return 0
  fi

  need_cmd curl
  need_cmd tar
  local tarball="$tmpdir/upstream.tar.gz"
  local extract_dir="$tmpdir/extract"
  local archive_url="https://github.com/HansZ8/RoboJuDo/archive/refs/heads/${UPSTREAM_REF}.tar.gz"

  echo "  git/rsync not available; falling back to tarball download"
  curl -fsSL -o "$tarball" "$archive_url"
  mkdir -p "$extract_dir"
  tar -xzf "$tarball" -C "$extract_dir"
  local src="$extract_dir/RoboJuDo-${UPSTREAM_REF}/assets"
  [[ -d "$src" ]] || {
    echo "ERROR: assets/ not found in upstream tarball" >&2
    exit 1
  }
  mkdir -p "$dest"
  cp -a "$src/." "$dest/"
  echo "  ok assets/ (tarball)"
}

apply_extras_zip() {
  local tmpdir="$1"

  if [[ "$SKIP_EXTRAS" == "1" ]]; then
    echo "==> Skipping semantic-WBC extras (ROBOJUDO_SKIP_EXTRAS=1)"
    return 0
  fi

  need_cmd curl
  need_cmd unzip

  local zip_path="$tmpdir/$EXTRAS_ZIP_NAME"
  echo "==> Trying semantic-WBC extras: $EXTRAS_URL"

  if ! curl -fsSL -o "$zip_path" "$EXTRAS_URL"; then
    echo "  extras zip not available (optional) — add later with ROBOJUDO_ASSETS_URL=file:///path/to/zip ./install_assets.sh"
    return 0
  fi

  unzip -q "$zip_path" -d "$tmpdir/extras"
  local src="$tmpdir/extras"
  [[ -d "$tmpdir/extras/release/assets" ]] && src="$tmpdir/extras/release"
  [[ -d "$tmpdir/extras/assets" ]] || {
    echo "ERROR: bad extras zip layout (expected assets/)" >&2
    exit 1
  }
  src="$src"

  for dir in assets mp3_songs shazam; do
    [[ -d "$src/$dir" ]] || continue
    mkdir -p "$ROOT/$dir"
    if command -v rsync >/dev/null 2>&1; then
      rsync -a "$src/$dir/" "$ROOT/$dir/"
    else
      cp -a "$src/$dir/." "$ROOT/$dir/"
    fi
    echo "  ok $dir/ (extras overlay)"
  done

  if [[ -f "$src/demo/real_try_trimmed_minus2s.mp3" ]]; then
    mkdir -p "$ROOT/demo"
    cp -f "$src/demo/real_try_trimmed_minus2s.mp3" "$ROOT/demo/"
    cp -f "$src/demo/real_try_trimmed_minus2s.mp3" "$ROOT/" 2>/dev/null || true
    echo "  ok demo/real_try_trimmed_minus2s.mp3"
  fi
}

report_status() {
  local missing_onnx=()
  local onnx

  for onnx in "${SEMANTIC_ONNX[@]}"; do
    [[ -f "$ROOT/assets/models/g1/beyondmimic/$onnx" ]] || missing_onnx+=("$onnx")
  done

  echo
  echo "==> Asset status"
  if [[ -f "$ROOT/assets/models/g1/unitree/policy_lstm_1.pt" && -f "$ROOT/assets/robots/g1/g1_29dof_rev_1_0.xml" ]]; then
    echo "  base RoboJuDo assets: OK (sim + g1_beyondmimic / g1_switch)"
  else
    echo "  base RoboJuDo assets: INCOMPLETE — re-run ./install_assets.sh"
  fi

  if ((${#missing_onnx[@]} == 0)); then
    echo "  semantic-WBC demo ONNX: OK (g1_switch_beyondmimic / music demos)"
  else
    echo "  semantic-WBC demo ONNX: missing ${#missing_onnx[@]} file(s)"
    printf '    - %s\n' "${missing_onnx[@]}"
    echo "    Drop them under assets/models/g1/beyondmimic/ or install an extras zip."
  fi

  if compgen -G "$ROOT/mp3_songs/*.mp3" >/dev/null; then
    echo "  mp3_songs: OK"
  else
    echo "  mp3_songs: missing (needed for music/Shazam demos)"
  fi

  if [[ -f "$ROOT/shazam/index.pkl" ]]; then
    echo "  shazam/index.pkl: OK"
  else
    echo "  shazam/index.pkl: missing (rebuild from mp3_songs/ or add via extras zip)"
  fi

  echo
  if ((${#missing_onnx[@]} == 0)); then
    echo "Done. Test: python scripts/run_pipeline.py -c g1_switch_beyondmimic"
  elif [[ -f "$ROOT/assets/models/g1/beyondmimic/Jump_wose.onnx" ]]; then
    echo "Done. Base sim ready: python scripts/run_pipeline.py -c g1_beyondmimic"
  else
    echo "Done. Install finished; check missing items above."
  fi
}

main() {
  local tmpdir
  tmpdir="$(mktemp -d)"
  # shellcheck disable=SC2064
  trap "rm -rf '$tmpdir'" EXIT

  fetch_upstream_assets "$tmpdir"
  apply_extras_zip "$tmpdir"
  report_status
}

main "$@"
