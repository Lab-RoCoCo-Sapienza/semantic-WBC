#!/usr/bin/env bash
# Build assets zip for GitHub Release (run from standalone release/ or full repo).
set -euo pipefail

REL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "$REL/robojudo" ]]; then
  ROOT="$REL"
elif [[ -d "$REL/../robojudo" ]]; then
  ROOT="$(cd "$REL/.." && pwd)"
else
  echo "ERROR: cannot find project root." >&2
  exit 1
fi

STAGE="$REL/dist/staging"
DIST="$REL/dist"
VERSION="$(grep -E '^version\s*=' "$ROOT/pyproject.toml" | head -1 | sed 's/.*"\(.*\)".*/\1/')"
ZIP="$DIST/robojudo-assets-v${VERSION}.zip"

BEYOND=(
  bts_2_0_tracking_2.onnx bts_dynamite_tracking.onnx easy_sample.onnx
  Swim_tracking.onnx thriller.onnx salsa_tracking.onnx gdance.onnx Salsa_4.onnx
  thriller_locked_waist.onnx g1_29dof_gesti_10k.onnx g1_29dof_67_10k.onnx
  g1_29dof_67_safe.onnx g1_29dof_gesti_safe.onnx Violin.onnx
)

echo "Staging assets v${VERSION}..."
rm -rf "$STAGE"
mkdir -p "$STAGE/assets/models/g1/beyondmimic" "$STAGE/assets/models/g1/unitree"
mkdir -p "$STAGE/assets/models" "$STAGE/assets/robots/g1"
mkdir -p "$STAGE/mp3_songs" "$STAGE/shazam" "$STAGE/demo"

for f in "${BEYOND[@]}"; do
  cp -f "$ROOT/assets/models/g1/beyondmimic/$f" "$STAGE/assets/models/g1/beyondmimic/"
done
cp -f "$STAGE/assets/models/g1/beyondmimic/g1_29dof_gesti_10k.onnx" \
  "$STAGE/assets/models/g1/beyondmimic/g1_29dof_gesti.onnx"

cp -f "$ROOT/assets/models/g1/unitree/policy_lstm_1.pt" \
      "$ROOT/assets/models/g1/unitree/policy_wo_gait.pt" \
  "$STAGE/assets/models/g1/unitree/"

cp -f "$ROOT/assets/models/rhythm_classifier.joblib" \
      "$ROOT/assets/models/rhythm_classifier.meta.json" \
  "$STAGE/assets/models/" 2>/dev/null || true

rsync -a "$ROOT/assets/robots/g1/" "$STAGE/assets/robots/g1/"
rsync -a "$ROOT/mp3_songs/" "$STAGE/mp3_songs/"

for f in index.pkl clap_index.json clap_index.npy; do
  [[ -f "$ROOT/shazam/$f" ]] || { echo "ERROR: missing shazam/$f" >&2; exit 1; }
  cp -f "$ROOT/shazam/$f" "$STAGE/shazam/"
done

[[ -f "$ROOT/real_try_trimmed_minus2s.mp3" ]] && \
  cp -f "$ROOT/real_try_trimmed_minus2s.mp3" "$STAGE/demo/"
[[ -f "$ROOT/demo/real_try_trimmed_minus2s.mp3" ]] && \
  cp -f "$ROOT/demo/real_try_trimmed_minus2s.mp3" "$STAGE/demo/"

mkdir -p "$DIST"
rm -f "$ZIP"
( cd "$STAGE" && zip -r -q "$ZIP" assets mp3_songs shazam demo )

echo "Built: $ZIP ($(du -sh "$ZIP" | awk '{print $1}'))"
