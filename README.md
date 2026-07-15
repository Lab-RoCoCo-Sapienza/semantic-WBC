# RoboJuDo — release standalone

Cartella **autocontenuta** da pubblicare su GitHub: contiene **codice + documentazione**, non i binari pesanti (ONNX, mp3, mesh).

Puoi clonare **solo questa cartella** (o il repo che la espone) su **robot e PC** — stesso contenuto, script diversi.

## Cosa c'è dentro (codice)

| Path | Cosa fa |
|------|---------|
| `robojudo/` | Framework pipeline, config G1, policy, controller, env |
| `scripts/run_pipeline_safe.py` | **Robot** — esecuzione su G1 reale |
| `scripts/run_pipeline.py` | **Sim** — MuJoCo |
| `scripts/listen_smart_mic_timed_policy.py` | **PC** — listener audio → TCP |
| `scripts/listen_smart_mic.py` | Dipendenza del listener |
| `scripts/listen_shazam_and_send.py` | Listener Shazam alternativo |
| `shazam/` | Matcher locale (+ indici scaricati con `install_assets.sh`) |
| `listen_openai_and_send_gesti/` | Listener voce → `[GESTI,sec]` |
| `packages/unitree_cpp/` | Binding robot (solo onboard G1) |
| `third_party/mujoco_viewer/` | Viewer sim |
| `pyproject.toml`, `requirements.txt` | Install Python |

## Cosa NON c'è in git (download separato)

ONNX, `.pt`, mp3, mesh STL, `shazam/index.pkl` → **GitHub Release**:

```bash
./install_assets.sh
```

Vedi [MANIFEST.md](MANIFEST.md).

---

## Installazione rapida (robot o PC)

```bash
git clone <URL-REPO-RELEASE> RoboJuDo
cd RoboJuDo

python3 -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -r requirements-listener.txt   # solo sul PC listener

# Robot onboard:
python submodule_install.py unitree_cpp

# Binari (ONNX, audio, mesh):
./install_assets.sh
```

---

## Cosa lanciare

### Robot (G1 PC2)

```bash
python scripts/run_pipeline_safe.py \
  -c g1_gesti_multi_real \
  --cmd-server-host <IP_PC> \
  --cmd-server-port 8765 \
  --cmd-subscribe-topic gesture \
  --cmd-duration-unit seconds
```

### PC (listener)

```bash
python scripts/listen_smart_mic_timed_policy.py \
  --verbose --tcp-mode server --server-host 0.0.0.0 --server-port 8765 \
  --clap-fallback --music-only \
  --offline-mashup demo/real_try_trimmed_minus2s.mp3 \
  --offline-playback \
  --song-to-policy "thriller:4,salsa:2,dynamite:3"
```

Stesso repo su entrambe le macchine. Dettagli: [DEPLOY.md](DEPLOY.md).

---

## Documentazione

| File | Contenuto |
|------|-----------|
| [DEPLOY.md](DEPLOY.md) | Robot vs PC, stesso repo |
| [PIPELINE.md](PIPELINE.md) | Tutti i comandi (sim, Docker, real) |
| [MANIFEST.md](MANIFEST.md) | Asset esterni, cosa serve davvero |

---

## Maintainer (repo principale)

Dopo modifiche al codice nel repo full:

```bash
./release/sync_deploy_code.sh    # copia codice aggiornato in release/
./release/package_assets.sh      # zip ONNX/audio → release/dist/
```

Poi push `release/` su GitHub e allega `release/dist/*.zip` alla Release.
