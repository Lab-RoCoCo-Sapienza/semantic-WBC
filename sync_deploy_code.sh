#!/usr/bin/env bash
# Copy minimal runnable code from repo root into release/ (standalone deploy bundle).
# Run from repo root: ./release/sync_deploy_code.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Syncing deploy code: $ROOT -> $REL"

copy_file() {
  local src="$1" dst="$2"
  mkdir -p "$(dirname "$dst")"
  cp -f "$src" "$dst"
}

rsync_dir() {
  rsync -a --delete "$1" "$2"
}

# --- Root install files ---
for f in pyproject.toml requirements.txt submodule_install.py submodule_cfg.yaml; do
  copy_file "$ROOT/$f" "$REL/$f"
done

# --- Full robojudo package (config registry imports h1; ship whole tree) ---
rsync_dir "$ROOT/robojudo/" "$REL/robojudo/"

# --- Essential scripts only ---
mkdir -p "$REL/scripts"
for f in \
  run_pipeline.py \
  run_pipeline_safe.py \
  listen_smart_mic.py \
  listen_smart_mic_timed_policy.py \
  listen_shazam_and_send.py \
  stream_shazam_source.py \
  rhythm_features.py
do
  copy_file "$ROOT/scripts/$f" "$REL/scripts/$f"
done

# --- Shazam runtime (indices via install_assets.sh) ---
mkdir -p "$REL/shazam"
for f in local_fingerprint.py clap_fallback.py _normalize.py; do
  copy_file "$ROOT/shazam/$f" "$REL/shazam/$f"
done

# --- Optional OpenAI gesti listener ---
rsync_dir "$ROOT/listen_openai_and_send_gesti/" "$REL/listen_openai_and_send_gesti/"

# --- Native deps (submodules) ---
if [[ -d "$ROOT/third_party/mujoco_viewer" ]]; then
  rsync_dir "$ROOT/third_party/mujoco_viewer/" "$REL/third_party/mujoco_viewer/"
fi
if [[ -d "$ROOT/third_party/patches" ]]; then
  rsync_dir "$ROOT/third_party/patches/" "$REL/third_party/patches/"
fi
if [[ -d "$ROOT/packages/unitree_cpp" ]]; then
  rsync_dir "$ROOT/packages/unitree_cpp/" "$REL/packages/unitree_cpp/"
fi

# --- Listener extras not in base requirements.txt ---
cat > "$REL/requirements-listener.txt" <<'EOF'
# PC listener extras (install after: pip install -e .)
sounddevice
transformers
openai
EOF

# --- Placeholder dirs for install_assets.sh ---
mkdir -p "$REL/assets" "$REL/mp3_songs" "$REL/demo"
touch "$REL/assets/.gitkeep" "$REL/mp3_songs/.gitkeep" "$REL/demo/.gitkeep"

find "$REL" -name '.DS_Store' -delete 2>/dev/null || true
find "$REL" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

echo ""
echo "Standalone release/ updated."
echo "  Python:  $(find "$REL/robojudo" -name '*.py' | wc -l | tr -d ' ') files in robojudo/"
echo "  Scripts: $(ls "$REL/scripts" | wc -l | tr -d ' ')"
du -sh "$REL" | awk '{print "  Total:   " $1}'
