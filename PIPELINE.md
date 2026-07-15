# RoboJuDo — Release Pipeline

End-to-end command reference for installing RoboJuDo and running the demo pipelines: local simulation, music-driven policy switching, split PC/robot deploy, and real Unitree G1.

All commands assume you are at the **repository root** unless noted otherwise.

```bash
export ROBOJUDO_ROOT="$(pwd)"
cd "$ROBOJUDO_ROOT"
```

---

## Table of contents

1. [Overview](#1-overview)
2. [System requirements](#2-system-requirements)
3. [Installation](#3-installation)
4. [One-time audio setup (Shazam index)](#4-one-time-audio-setup-shazam-index)
5. [Pipeline A — Basic MuJoCo simulation](#5-pipeline-a--basic-mujoco-simulation)
6. [Pipeline B — Sim + file-based Shazam (chunk watch)](#6-pipeline-b--sim--file-based-shazam-chunk-watch)
7. [Pipeline C — Sim + TCP listener (live mic)](#7-pipeline-c--sim--tcp-listener-live-mic)
8. [Pipeline D — Full demo (offline mashup + timed policies)](#8-pipeline-d--full-demo-offline-mashup--timed-policies)
9. [Pipeline E — Real Unitree G1 robot](#9-pipeline-e--real-unitree-g1-robot)
10. [Pipeline F — Docker (GPU workstation)](#10-pipeline-f--docker-gpu-workstation)
11. [Config quick reference](#11-config-quick-reference)
12. [Environment variables](#12-environment-variables)
13. [Troubleshooting](#13-troubleshooting)
14. [Further reading](#14-further-reading)

---

## 1. Overview

RoboJuDo runs a **multi-policy RL pipeline** on Unitree G1 (simulation or real). Music/audio listeners identify songs and send **TCP commands** (`[POLICY_SWITCH],N` or `[GESTI,seconds]`) to switch BeyondMimic motion policies.

Typical split setup:

```text
┌─────────────────────┐         TCP :8765          ┌──────────────────────┐
│  PC / listener      │  ───────────────────────►  │  Sim or G1 robot     │
│  (mic or offline    │   [POLICY_SWITCH],N        │  run_pipeline*.py    │
│   mashup audio)     │   [GESTI,1.25]             │                      │
└─────────────────────┘                            └──────────────────────┘
```

**Recommended order for a first run:**

1. Install Python env + dependencies
2. Build Shazam fingerprint index
3. Pipeline A (keyboard sim) — verify MuJoCo + ONNX models
4. Pipeline D (offline mashup demo) — full music→policy loop without a mic
5. Pipeline C or E — live mic or real robot (when hardware is ready)

---

## 2. System requirements

### All pipelines

| Requirement | Notes |
|-------------|-------|
| Python | **3.10+** (3.11/3.12 tested; 3.13 needs `audioop-lts` from `requirements.txt`) |
| Git | For clone and optional submodules |
| `ffmpeg` | Audio chunking, Shazam normalization |
| GPU (recommended) | CUDA for CLAP/transformers; CPU works for basic sim |

### Simulation only

| Requirement | Notes |
|-------------|-------|
| MuJoCo | Installed via `pip` (`requirements.txt`) |
| Display | Local GUI, or Docker + X11 (Pipeline F) |

### Real G1 robot

| Requirement | Notes |
|-------------|-------|
| Unitree G1 PC2 (onboard Linux) | Recommended runtime host |
| [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2) (C++) | Install **before** `unitree_cpp` |
| `unitree_cpp` | `python submodule_install.py unitree_cpp` |
| Ethernet | Robot interface usually `eth0` |
| Debug/Developer mode | Enable from Unitree app before deploy |

### Audio listener (live mic)

| Requirement | Notes |
|-------------|-------|
| PortAudio | `sounddevice` (in `requirements.txt`) |
| Microphone | Or `--offline-mashup` for reproducible demos |
| `OPENAI_API_KEY` | Only if using speech/GESTI path in `listen_smart_mic_timed_policy.py` |

---

## 3. Installation

### 3.0 External assets (not in git)

Base RoboJuDo assets come from upstream; semantic-WBC demo extras are optional:

```bash
cd semantic-WBC
./install_assets.sh
```

Optional overlay when you have the extras zip:

```bash
ROBOJUDO_ASSETS_URL="file:///path/to/robojudo-assets-v1.5.0.zip" ./install_assets.sh
```

See `MANIFEST.md` and `assets_urls.yaml` for **what is required vs optional** per pipeline (sim-only, real robot, music demo).

### 3.1 Clone

```bash
git clone <REPO_URL> RoboJuDo
cd RoboJuDo
./install_assets.sh   # if assets not downloaded yet
```

If the repo uses submodules:

```bash
git submodule update --init --recursive
```

### 3.2 Python virtual environment

```bash
python3 -m venv .venv_robojudo
source .venv_robojudo/bin/activate   # Windows: .venv_robojudo\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
```

### 3.3 Install RoboJuDo and dependencies

**CPU / macOS (default PyPI wheels):**

```bash
pip install -e .
python submodule_install.py mujoco_viewer
```

**NVIDIA GPU (Linux, CUDA 12.8 example):**

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
pip install onnxruntime-gpu
pip install -e .
python submodule_install.py mujoco_viewer
```

**Real robot only — add Unitree C++ binding:**

```bash
# 1) Install unitree_sdk2 first (follow Unitree official docs)
# 2) Then:
python submodule_install.py unitree_cpp
```

Verify imports:

```bash
# Sim
python -c "from robojudo.environment import MujocoEnv; print('MuJoCo OK')"

# Real robot (on PC2)
python -c "from robojudo.environment import UnitreeCppEnv; print('UnitreeCpp OK')"
```

### 3.4 Optional extras

```bash
# Dev tools
pip install -e ".[dev]"

# OpenAI speech/GESTI listener
pip install openai

# Real Shazam cross-check (optional, needs internet)
pip install shazamio
```

---

## 4. One-time audio setup (Shazam index)

Required for any music-driven pipeline (B, C, D, E).

Source songs live in `mp3_songs/` (butter, dynamite, swim, salsa, thriller, etc.).

```bash
source .venv_robojudo/bin/activate
cd "$ROBOJUDO_ROOT"

# Fingerprint index (local Shazam-style matcher)
python shazam/run_experiment.py --build-index \
  --songs-dir mp3_songs \
  --index-path shazam/index.pkl
```

**Optional but recommended — CLAP fallback** (when fingerprint votes are weak):

```bash
python shazam/run_experiment.py --build-clap-index \
  --songs-dir mp3_songs \
  --clap-index shazam/clap_index.json
```

Quick sanity check on one clip:

```bash
python shazam/run_experiment.py \
  --index-path shazam/index.pkl \
  --query mp3_songs/"BTS - Dynamite (Lyrics).mp3" \
  --clap-fallback --clap-index shazam/clap_index.json
```

---

## 5. Pipeline A — Basic MuJoCo simulation

**Goal:** Verify sim, policies, and keyboard control. No audio.

**Config:** `g1_switch_beyondmimic`

```bash
source .venv_robojudo/bin/activate
cd "$ROBOJUDO_ROOT"

python scripts/run_pipeline.py -c g1_switch_beyondmimic
```

**Hotkeys (MuJoCo HUD):**

| Key | Action |
|-----|--------|
| `w` | Walk (policy 0) |
| `s` | Stand (policy 1) |
| `1`–`9` | BeyondMimic policies 2–10 |
| `Tab` | Cycle policies |
| `r` | Reborn (reset sim + return to Walk) |
| `Esc` | Quit |

**Variant with SAFE ONNX snapshots:** `g1_switch_beyondmimic_gesti_safe`

---

## 6. Pipeline B — Sim + file-based Shazam (chunk watch)

**Goal:** Pre-recorded audio chunks on disk drive policy switches (no TCP).

**Config:** `g1_shazam_beyondmimic`  
**Watch directory:** `runtime_chunks/` (must match stream script `--out-dir`)

### Terminal 1 — simulation

```bash
source .venv_robojudo/bin/activate
cd "$ROBOJUDO_ROOT"

python scripts/run_pipeline.py -c g1_shazam_beyondmimic
```

### Terminal 2 — stream chunks into watch folder

```bash
source .venv_robojudo/bin/activate
cd "$ROBOJUDO_ROOT"

python scripts/stream_shazam_source.py --song swim --out-dir runtime_chunks
```

Other songs: `butter`, `salsa`, `swim`, `dynamite`.

Options:

- `--no-clean` — do not delete old chunks before streaming
- `--segment-seconds 5` — chunk length (default 5 s)

---

## 7. Pipeline C — Sim + TCP listener (live mic)

**Goal:** Microphone → song ID → TCP → sim switches policy.

**Sim config:** `g1_shazam_remote_listener` (listens on TCP **8765**)

### Terminal 1 — simulation (server)

```bash
source .venv_robojudo/bin/activate
cd "$ROBOJUDO_ROOT"

python scripts/run_pipeline.py -c g1_shazam_remote_listener
```

### Terminal 2 — live mic listener (client)

```bash
source .venv_robojudo/bin/activate
cd "$ROBOJUDO_ROOT"

python scripts/listen_shazam_and_send.py \
  --client-host 127.0.0.1 \
  --client-port 8765 \
  --song-to-policy "dynamite:3,swim:5,salsa:7,thriller:6,salsa4:9,gdance:8,bts:2"
```

**Dry-run** (no TCP, log matches only):

```bash
python scripts/listen_shazam_and_send.py --dry-run -v
```

**List microphones:**

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
python scripts/listen_shazam_and_send.py --sd-device 1 --dry-run -v
```

Policy IDs for `g1_shazam_remote_listener`:

| ID | Policy |
|----|--------|
| 0 | Stand |
| 2 | salsa_tracking |
| 3 | bts_dynamite_tracking |
| 4 | thriller |

---

## 8. Pipeline D — Full demo (offline mashup + timed policies)

**Goal:** Reproducible end-to-end demo without a microphone — offline mashup plays on speakers while the listener sends timed policy commands. This is the **recommended public demo** path.

### Architecture

```text
Terminal 1 (sim/robot)     Terminal 2 (listener, TCP server)
run_pipeline*.py      ◄──  listen_smart_mic_timed_policy.py
                         --offline-mashup ... --offline-playback
                         --tcp-mode server --server-port 8765
```

### D.1 Simulation demo (two terminals, same machine)

**Terminal 1:**

```bash
source .venv_robojudo/bin/activate
cd "$ROBOJUDO_ROOT"

python scripts/run_pipeline.py -c g1_shazam_remote_listener
```

**Terminal 2:**

```bash
source .venv_robojudo/bin/activate
cd "$ROBOJUDO_ROOT"

python scripts/listen_smart_mic_timed_policy.py \
  --verbose \
  --tcp-mode server \
  --server-host 0.0.0.0 \
  --server-port 8765 \
  --clap-fallback \
  --music-only \
  --offline-mashup path/to/your_mashup.mp3 \
  --offline-playback \
  --song-to-policy "thriller:4,salsa:2,dynamite:3" \
  --command-delay-seconds 0.7 \
  --final-policy-id 0 \
  --final-policy-repeat 1 \
  --final-policy-settle-seconds 2
```

Adjust `--song-to-policy` to match your config’s policy IDs. For `g1_shazam_remote_listener`, use IDs **2, 3, 4** (not 4, 2, 3 as in the real-robot gesti mapping).

### D.2 Split demo — sim on GPU laptop, listener on PC

**On sim machine** (replace `LISTENER_IP` with the listener host):

```bash
export ROBOJUDO_CMD_SERVER_HOST=LISTENER_IP
export ROBOJUDO_CMD_SERVER_PORT=8765
export ROBOJUDO_CMD_SUBSCRIBE_TOPIC=gesture

python scripts/run_pipeline.py -c g1_shazam_remote_listener_headless
```

**On listener machine:**

```bash
python scripts/listen_smart_mic_timed_policy.py \
  --verbose \
  --tcp-mode server \
  --server-host 0.0.0.0 \
  --server-port 8765 \
  --clap-fallback \
  --music-only \
  --offline-mashup path/to/your_mashup.mp3 \
  --offline-playback \
  --song-to-policy "thriller:4,salsa:2,dynamite:3" \
  --command-delay-seconds 0.7 \
  --final-policy-id 0 \
  --final-policy-repeat 1 \
  --final-policy-settle-seconds 2
```

`g1_shazam_remote_listener_headless` connects **outbound** to the listener when `ROBOJUDO_CMD_SERVER_HOST` is set (sim acts as TCP client).

### D.3 OpenAI speech → GESTI (optional)

Separate script for voice-driven gesture timing:

```bash
export OPENAI_API_KEY="sk-..."

python listen_openai_and_send_gesti/listen_openai_and_send_gesti.py \
  --tcp-mode server \
  --server-host 0.0.0.0 \
  --server-port 8765 \
  --segment-seconds 5
```

Sends lines like `[GESTI,1.25]` over TCP.

---

## 9. Pipeline E — Real Unitree G1 robot

**Goal:** G1 onboard runs policies; PC runs audio listener.

> **Safety:** Read `docs/g1_real_safe_startup.md` before any real-robot run. Use Debug mode, clear area, test E-stop first.

### 9.1 One-time setup on G1 PC2

```bash
cd ~/RoboJuDo
source .venv_robojudo/bin/activate

# After unitree_sdk2 is installed:
python submodule_install.py unitree_cpp
```

Confirm in `robojudo/config/g1/g1_custom_cfg.py` → `g1_gesti_multi_real`:

- `env_type="UnitreeCppEnv"`
- `net_if="eth0"` (or your interface)

### 9.2 Network layout

| Host | IP (example) | Role |
|------|----------------|------|
| G1 PC2 | `192.168.88.224` | Robot pipeline (TCP **client** → PC) |
| Demo PC | `192.168.88.100` | Listener (TCP **server** :8765) |

Replace with your LAN addresses.

### 9.3 Start robot pipeline (G1 PC2)

```bash
source .venv_robojudo/bin/activate
cd ~/RoboJuDo

python scripts/run_pipeline_safe.py \
  -c g1_gesti_multi_real \
  --cmd-server-host 192.168.88.100 \
  --cmd-server-port 8765 \
  --cmd-subscribe-topic gesture \
  --cmd-duration-unit seconds
```

**Debug TCP only** (no robot motion):

```bash
python scripts/run_pipeline_safe.py \
  -c g1_gesti_multi_real \
  --cmd-server-host 192.168.88.100 \
  --cmd-server-port 8765 \
  --cmd-subscribe-topic gesture \
  --cmd-duration-unit seconds \
  --cmd-debug-only
```

### 9.4 Start listener (demo PC)

```bash
source .venv_robojudo/bin/activate
cd "$ROBOJUDO_ROOT"

python scripts/listen_smart_mic_timed_policy.py \
  --verbose \
  --tcp-mode server \
  --server-host 0.0.0.0 \
  --server-port 8765 \
  --clap-fallback \
  --music-only \
  --offline-mashup path/to/your_mashup.mp3 \
  --offline-playback \
  --song-to-policy "thriller:4,salsa:2,dynamite:3" \
  --final-policy-id 0 \
  --final-policy-repeat 1 \
  --final-policy-settle-seconds 2
```

### 9.5 `g1_gesti_multi_real` policy map

| ID | Policy | Joystick |
|----|--------|----------|
| 0 | Stand (fallback) | `B` |
| 1 | g1_29dof_gesti | `X` / `[GESTI,*]` |
| 2 | salsa_tracking | `Up` |
| 3 | bts_dynamite_tracking | `Left` |
| 4 | thriller | `Right` |
| 5 | g1_29dof_67_10k | `Down` / `[67]` |
| 6 | Violin | `L1` / `[Violin]` |

Remote TCP tokens: `[GESTI,seconds]`, `[Violin]`, `[67]`, plus `[POLICY_SWITCH],N`.

### 9.6 Safe shutdown

1. `Ctrl+C` on robot pipeline terminal
2. E-stop on controller if needed
3. Exit Debug mode from Unitree app

---

## 10. Pipeline F — Docker (GPU workstation)

**Goal:** CUDA 12.8 container with MuJoCo GUI via X11 (Linux host).

### 10.1 Build image

```bash
cd "$ROBOJUDO_ROOT"

docker build -t robojudo:cu128-runtime .
```

Optional PHC motionlib:

```bash
docker build --build-arg INSTALL_PHC=1 -t robojudo:cu128-runtime .
```

### 10.2 Run container (helper script)

```bash
./scripts/run_robojudo_docker.sh robojudo:cu128-runtime
```

Inside the container, repo is mounted at `/workspace/RoboJuDo`. Then run any pipeline, e.g.:

```bash
cd /workspace/RoboJuDo
python scripts/run_pipeline.py -c g1_switch_beyondmimic
```

### 10.3 Manual `docker run` (with PulseAudio)

If you need speaker output from the container:

```bash
xhost +local:docker

docker run --rm -it --gpus all \
  --shm-size=4g \
  --device /dev/dri \
  -e DISPLAY=$DISPLAY \
  -e XAUTHORITY=$XAUTHORITY \
  -e MUJOCO_GL=glfw \
  -e SDL_AUDIODRIVER=pulse \
  -e PULSE_SERVER=unix:${XDG_RUNTIME_DIR}/pulse/native \
  -e __GLX_VENDOR_LIBRARY_NAME=nvidia \
  -e __NV_PRIME_RENDER_OFFLOAD=1 \
  -e __VK_LAYER_NV_optimus=NVIDIA_only \
  -e WAYLAND_DISPLAY= \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$XAUTHORITY":"$XAUTHORITY" \
  -v "${XDG_RUNTIME_DIR}/pulse/native:${XDG_RUNTIME_DIR}/pulse/native" \
  -v "$PWD":/workspace/RoboJuDo \
  robojudo:cu128-runtime
```

---

## 11. Config quick reference

| Config | Environment | Purpose |
|--------|-------------|---------|
| `g1_switch_beyondmimic` | MuJoCo sim | Keyboard multi-policy demo |
| `g1_switch_beyondmimic_gesti_safe` | MuJoCo sim | Gesti/67 with SAFE ONNX hotkeys |
| `g1_shazam_beyondmimic` | MuJoCo sim | File-watch Shazam → policy |
| `g1_shazam_remote_listener` | MuJoCo sim | TCP server for remote listener |
| `g1_shazam_remote_listener_headless` | MuJoCo sim | No keyboard; TCP client or server |
| `g1_gesti_multi_real` | Real G1 | Multi-policy + remote TCP + joystick |
| `g1_real` | Real G1 | Basic real deploy (see unitree docs) |

List all registered configs:

```bash
python -c "from robojudo.config.config_manager import ConfigManager; print('Use -c <name> with scripts/run_pipeline.py')"
```

---

## 12. Environment variables

| Variable | Used by | Effect |
|----------|---------|--------|
| `ROBOJUDO_CMD_SERVER_HOST` | `g1_shazam_remote_listener_headless` | Listener IP (sim connects as client) |
| `ROBOJUDO_CMD_SERVER_PORT` | headless config | TCP port (default `8765`) |
| `ROBOJUDO_CMD_SUBSCRIBE_TOPIC` | headless / `run_pipeline_safe` | Subscribe topic (default `gesture`) |
| `ROBOJUDO_MUSIC_PORT` | `g1_shazam_remote_listener` | TCP listen port (default `8765`) |
| `ROBO_SHAZAM_WATCH_DIR` | `stream_shazam_source.py` | Override chunk output directory |
| `ROBO_SHAZAM_CAPTURE` | `listen_shazam_and_send.py` | `sounddevice` or `ffmpeg` |
| `ROBO_SHAZAM_MIC` | `listen_shazam_and_send.py` | ffmpeg AVFoundation device string |
| `OPENAI_API_KEY` | smart mic / OpenAI gesti | Speech and TTS |
| `MUJOCO_GL` | MuJoCo | `glfw` (GUI), `egl` (headless) |

---

## 13. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ImportError: unitree_cpp` | Install `unitree_sdk2`, then `python submodule_install.py unitree_cpp` |
| MuJoCo window does not open / `Failed to open display :0` | Check `ls /tmp/.X11-unix/` (`X1` → `export DISPLAY=:1`); or `ROBOJUDO_HEADLESS=1` + `g1_shazam_remote_listener_headless`; Docker: X11 + `xhost +local:docker` |
| Shazam always wrong / low votes | Rebuild index; use `--segment-seconds 10`, `--min-votes 10`, `--clap-fallback` |
| TCP connection refused | Start listener **before** sim; check firewall on port 8765 |
| Policy IDs do not match song | Align `--song-to-policy` with the **active config** policy list |
| macOS mic permission | Grant microphone access to Terminal/Python in System Settings |
| Real robot frame drop | Use `UnitreeCppEnv`; reduce load; check `net_if` |
| Silent chunks skipped | Lower `--skip-if-peak-below-dbfs` or increase speaker volume |

---

## 14. Further reading

| Document | Topic |
|----------|-------|
| `README.md` | Current sim setup notes |
| `docs/g1_real_safe_startup.md` | Real robot safety checklist |
| `docs/unitree_setup.md` | Unitree SDK install |
| `docs/local_shazam.md` | Shazam matcher + streaming architecture |
| `listen_shazam_and_send/README.md` | TCP Shazam listener |
| `listen_openai_and_send_gesti/README.md` | OpenAI GESTI listener |
| `shazam/README.md` | Index build and evaluation |
| `metrics/README.md` | Benchmark and evaluation protocol |
| `COMANDI_ROBOJUDO.txt` | Quick command cheat sheet (internal IPs) |

---

## Quick copy-paste — minimum viable demo

```bash
# 1. Install
python3 -m venv .venv && source .venv/bin/activate
pip install -e . && python submodule_install.py mujoco_viewer
./install_assets.sh   # HansZ8 base + semantic-WBC extras zip when Release exists

# 2. Terminal 1 — sim (set DISPLAY if needed, e.g. export DISPLAY=:1)
python scripts/run_pipeline.py -c g1_shazam_remote_listener

# 4. Terminal 2 — offline mashup demo
python scripts/listen_smart_mic_timed_policy.py \
  --verbose --tcp-mode server --server-host 0.0.0.0 --server-port 8765 \
  --clap-fallback --music-only \
  --offline-mashup mp3_songs/"BTS - Dynamite (Lyrics).mp3" \
  --offline-playback \
  --song-to-policy "dynamite:3"
```

Replace the mashup path with your full demo mix when ready.
