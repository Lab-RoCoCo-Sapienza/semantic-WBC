# RoboJuDo — deployment overview (robot + PC)

**Short answer:** yes, it is the **same standalone release tree** on both the robot and the PC. Clone this folder (or repo) twice, install what each machine needs, run **different scripts**. No second codebase.

> This file lives inside the **standalone `release/` package**, which includes `robojudo/`, `scripts/`, etc. — not just shell scripts.

---

## Two machines, one repo

```text
                    SAME REPO: github.com/Lab-RoCoCo-Sapienza/semantic-WBC
                                    │
            ┌───────────────────────┴───────────────────────┐
            ▼                                               ▼
   G1 PC2 (robot onboard)                          Demo PC / laptop
   ─────────────────────                          ─────────────────
   git clone + pip install                        git clone + pip install
   + unitree_cpp                                  (no unitree_cpp)
   + ONNX models                                  + mp3 + Shazam index
   (no mic, no MuJoCo meshes)                     (no robot SDK)

   scripts/run_pipeline_safe.py                   scripts/listen_smart_mic_timed_policy.py
   -c g1_gesti_multi_real                         --tcp-mode server --server-port 8765
   --cmd-server-host <PC_IP>                      (+ mic or --offline-mashup)

            │                                               │
            └──────────── TCP :8765 ────────────────────────┘
                  robot connects TO the PC (client → server)
```

---

## What to install where

| | **Robot (G1 PC2)** | **PC (listener)** |
|---|-------------------|-------------------|
| **Clone repo** | yes | yes |
| **`pip install -e .`** | yes | yes |
| **`unitree_sdk2` + `unitree_cpp`** | yes | no |
| **BeyondMimic ONNX** | yes | no |
| **Unitree Walk/Stand `.pt`** | no | no (sim only) |
| **MuJoCo meshes** | no | only if you also run sim on PC |
| **`mp3_songs/` + Shazam index** | no* | yes (for music demo) |
| **Microphone / speakers** | no | yes |

\*You *can* run the listener on the robot, but the normal setup keeps audio on the PC.

### Minimal asset download per machine

**Robot only (real deploy):**

```bash
git clone <URL-REPO-RELEASE> RoboJuDo && cd RoboJuDo
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
python submodule_install.py unitree_cpp
./install_assets.sh
```

**PC only (listener + optional sim):**

```bash
git clone <URL-REPO-RELEASE> RoboJuDo && cd RoboJuDo
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -r requirements-listener.txt
./install_assets.sh
```

---

## Commands to run (real robot demo)

### 1 — Robot (G1 PC2)

Replace `192.168.88.100` with your **PC IP**.

```bash
source .venv/bin/activate
cd ~/RoboJuDo

python scripts/run_pipeline_safe.py \
  -c g1_gesti_multi_real \
  --cmd-server-host 192.168.88.100 \
  --cmd-server-port 8765 \
  --cmd-subscribe-topic gesture \
  --cmd-duration-unit seconds
```

### 2 — PC (start this **before** or at the same time as the robot)

```bash
source .venv/bin/activate
cd ~/RoboJuDo

python scripts/listen_smart_mic_timed_policy.py \
  --verbose \
  --tcp-mode server \
  --server-host 0.0.0.0 \
  --server-port 8765 \
  --clap-fallback \
  --music-only \
  --offline-mashup real_try_trimmed_minus2s.mp3 \
  --offline-playback \
  --song-to-policy "thriller:4,salsa:2,dynamite:3" \
  --final-policy-id 0 \
  --final-policy-repeat 1 \
  --final-policy-settle-seconds 2
```

Policy IDs **4 / 2 / 3** match `g1_gesti_multi_real` (thriller / salsa / dynamite). They differ from sim-only configs — see PIPELINE.md §9.5.

---

## Simulation only (one PC, no robot)

Same repo, **both** terminals on one machine — see [PIPELINE.md](PIPELINE.md) §5–§8.

---

## Which doc to read

| Document | Contents |
|----------|----------|
| **[DEPLOY.md](DEPLOY.md)** (this file) | Robot vs PC, same repo, what runs where |
| **[PIPELINE.md](PIPELINE.md)** | Full install → demo commands (sim, Docker, real robot) |
| **[MANIFEST.md](MANIFEST.md)** | External assets (ONNX, audio) — download links, required vs optional |
| **[README.md](README.md)** | Release folder index + quick start |
| **`docs/g1_real_safe_startup.md`** | Safety checklist before real robot |
| **`docs/unitree_setup.md`** | Unitree SDK install |
| **`COMANDI_ROBOJUDO.txt`** | Internal command cheat sheet |

**PIPELINE.md** is the main end-to-end guide; **this file** is the one-page answer for “same repo on robot and PC?”
