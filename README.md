# semantic-WBC

**Semantic Whole-Body Control for Unitree G1** — music and audio drive BeyondMimic motion policies over a split robot / PC pipeline.

Built on [RoboJuDo](https://github.com/HansZ8/RoboJuDo). This repository is the **standalone deploy bundle**: Python code, configs, and scripts. Base RoboJuDo assets are pulled from upstream on install; demo-specific ONNX and audio can be added via an optional extras zip (see [MANIFEST.md](MANIFEST.md)).

---

## Overview

| Component | Role |
|-----------|------|
| **Robot (G1 PC2)** | Runs multi-policy RL pipeline, executes ONNX motions, connects to PC over TCP |
| **PC (listener)** | Captures audio (mic or offline mashup), identifies songs, sends policy commands |
| **Simulation (optional)** | MuJoCo multi-policy demo on a workstation |

The same repository is cloned on both machines. Only the entry script differs.

```
┌──────────────────────┐      TCP :8765       ┌─────────────────────────┐
│  Unitree G1 (PC2)    │ ◄─────────────────── │  Demo PC                │
│  run_pipeline_safe   │   [POLICY_SWITCH],N  │  listen_smart_mic_*     │
│  BeyondMimic ONNX    │   [GESTI, seconds]   │  mic / offline mashup   │
└──────────────────────┘                      └─────────────────────────┘
```

---

## Requirements

| | Robot | PC listener | Sim only |
|---|:---:|:---:|:---:|
| Python ≥ 3.10 | ✓ | ✓ | ✓ |
| [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2) + `unitree_cpp` | ✓ | | |
| CUDA (recommended) | | ✓ | ✓ |
| `ffmpeg` | | ✓ | |
| Microphone / speakers | | ✓ | |

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/Lab-RoCoCo-Sapienza/semantic-WBC.git
cd semantic-WBC

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -e .
pip install -r requirements-listener.txt   # PC listener only
```

**Robot onboard** — after installing the Unitree C++ SDK:

```bash
python submodule_install.py unitree_cpp
```

### 2. Download assets

Binary files are **not** in git. `./install_assets.sh` does two steps:

1. **Base** (always) — sparse clone from [HansZ8/RoboJuDo](https://github.com/HansZ8/RoboJuDo) `release`: MuJoCo meshes, Walk/Stand `.pt`, Jump/Dance/Violin ONNX. Enough for `g1_beyondmimic` / `g1_switch`.
2. **Extras** (optional) — semantic-WBC demo zip: thriller/salsa/dynamite/gesti ONNX, `mp3_songs/`, Shazam index. Needed for `g1_switch_beyondmimic`, music demos, real gesti deploy.

```bash
python submodule_install.py mujoco_viewer   # sim viewer
./install_assets.sh
```

After a [GitHub Release](https://github.com/Lab-RoCoCo-Sapienza/semantic-WBC/releases) with `robojudo-assets-v1.5.0.zip` is published, step 2 downloads automatically. Until then:

```bash
# local zip or mirror
ROBOJUDO_ASSETS_URL="file:///path/to/robojudo-assets-v1.5.0.zip" ./install_assets.sh

# upstream only (no demo ONNX / no mp3)
ROBOJUDO_SKIP_EXTRAS=1 ./install_assets.sh
```

The script prints what is still missing. See [MANIFEST.md](MANIFEST.md) for the full inventory.

### 3. Run the demo

**Terminal 1 — robot (G1 PC2)**

```bash
python scripts/run_pipeline_safe.py \
  -c g1_gesti_multi_real \
  --cmd-server-host <PC_IP> \
  --cmd-server-port 8765 \
  --cmd-subscribe-topic gesture \
  --cmd-duration-unit seconds
```

**Terminal 2 — PC listener**

```bash
python scripts/listen_smart_mic_timed_policy.py \
  --verbose \
  --tcp-mode server \
  --server-host 0.0.0.0 \
  --server-port 8765 \
  --clap-fallback \
  --music-only \
  --offline-mashup demo/real_try_trimmed_minus2s.mp3 \
  --offline-playback \
  --song-to-policy "thriller:4,salsa:2,dynamite:3" \
  --final-policy-id 0
```

Replace `<PC_IP>` with the listener machine address. Start the listener before or together with the robot pipeline.

---

## Simulation (single machine)

If MuJoCo fails with `Failed to open display :0`, your X server may be on another display (e.g. `:1`):

```bash
export DISPLAY=:1   # adjust if needed; check ls /tmp/.X11-unix/
```

```bash
# Terminal 1
python scripts/run_pipeline.py -c g1_shazam_remote_listener

# Terminal 2
python scripts/listen_shazam_and_send.py \
  --client-host 127.0.0.1 \
  --client-port 8765 \
  --song-to-policy "dynamite:3,swim:5,salsa:7,thriller:6"
```

Keyboard sim without audio: `python scripts/run_pipeline.py -c g1_switch_beyondmimic`

---

## Repository layout

```
semantic-WBC/
├── robojudo/              # Pipeline, policies, controllers, G1 configs
├── scripts/               # Entry points (robot, sim, listener)
├── shazam/                # Local audio fingerprint matcher
├── packages/unitree_cpp/  # G1 onboard SDK binding
├── third_party/           # MuJoCo viewer (sim)
├── install_assets.sh      # Download ONNX / audio / meshes
├── DEPLOY.md              # Robot vs PC deployment
├── PIPELINE.md            # Full command reference
└── MANIFEST.md            # External assets inventory
```

---

## Policy mapping (`g1_gesti_multi_real`)

| ID | Motion | Remote / joystick |
|----|--------|-------------------|
| 0 | Stand | `B` |
| 1 | Gesti | `[GESTI,*]` / `X` |
| 2 | Salsa | `Up` |
| 3 | Dynamite | `Left` |
| 4 | Thriller | `Right` |
| 5 | 67 | `Down` / `[67]` |
| 6 | Violin | `L1` / `[Violin]` |

---

## Documentation

| Document | Description |
|----------|-------------|
| [DEPLOY.md](DEPLOY.md) | Two-machine setup, install matrix |
| [PIPELINE.md](PIPELINE.md) | Install, sim, Docker, troubleshooting |
| [MANIFEST.md](MANIFEST.md) | Asset bundle contents and download |

## Publishing assets (maintainers)

Binaries stay out of git (~120 MB zip). To ship them to users:

```bash
# with mp3_songs/, shazam indices, and demo ONNX already in the tree (see MANIFEST.md)
./package_assets.sh
gh release create v1.5.0 dist/robojudo-assets-v1.5.0.zip --title "Assets v1.5.0"
```

Users then run `./install_assets.sh` and receive base + extras automatically.

---

## Safety (real robot)

Before running on hardware:

1. Enable **Debug / Developer mode** on the G1.
2. Clear the workspace and verify **E-stop**.
3. Read the startup checklist in upstream [g1 real deploy docs](https://github.com/HansZ8/RoboJuDo/blob/release/docs/g1_real_safe_startup.md).

---

## Citation & credits

- **RoboJuDo** — [HansZ8/RoboJuDo](https://github.com/HansZ8/RoboJuDo)
- **Lab-RoCoCo, Sapienza University of Rome**

---

## License

See upstream RoboJuDo and bundled third-party packages (`packages/unitree_cpp`, `third_party/mujoco_viewer`) for their respective licenses.
