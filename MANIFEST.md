# RoboJuDo — external assets (not in git)

Large binaries are **not** committed to the repository. They are published as a **GitHub Release** zip and installed via link:

```bash
./release/install_assets.sh
```

Direct download (after you publish release `v1.5.0`):

**https://github.com/michelebri/RoboJuDo/releases/download/v1.5.0/robojudo-assets-v1.5.0.zip**

See [assets_urls.yaml](assets_urls.yaml) for the canonical URL config.

---

## Are they needed?

| Asset | Sim (keyboard) | Sim + music demo | Real G1 robot | Verdict |
|-------|----------------|------------------|---------------|---------|
| **BeyondMimic ONNX** (15 files) | yes | yes | yes | **Required** — policies cannot run without them |
| **Unitree Walk/Stand `.pt`** | yes | yes | no | **Required for sim** — Walk/Stand in multi-policy configs |
| **MuJoCo G1 XML + meshes** | yes | yes | no | **Required for sim** — not needed on real robot |
| **`mp3_songs/`** (7 tracks) | no | yes* | yes* | **Required for music demos** — or use your own songs |
| **`shazam/index.pkl` + CLAP index** | no | recommended | recommended | **Optional** — rebuild in ~2 min from `mp3_songs/` |
| **`real_try_trimmed_minus2s.mp3`** | no | optional | optional | **Optional** — any mashup/mp3 works with `--offline-mashup` |
| **`rhythm_classifier.joblib`** | no | no | no | **Optional** — only if using rhythm classifier flags |

\*Music demos need **some** audio source. Either download `mp3_songs/` or provide equivalent files and rebuild the Shazam index.

### Minimal installs

| Goal | Download | Extra steps |
|------|----------|-------------|
| Keyboard sim only | full zip (or models + mujoco subsets) | none |
| Real robot + remote listener | models ONNX only from zip | `unitree_cpp` on robot |
| Music → policy demo | full zip | none if using pre-built Shazam index |
| Music demo, no pre-built index | models + mujoco + `mp3_songs/` | run Shazam build commands below |

### Rebuild Shazam index (instead of downloading pre-built)

If you skip `shazam/index.pkl` from the release zip:

```bash
python shazam/run_experiment.py --build-index \
  --songs-dir mp3_songs --index-path shazam/index.pkl

python shazam/run_experiment.py --build-clap-index \
  --songs-dir mp3_songs --clap-index shazam/clap_index.json
```

---

## Zip contents (`robojudo-assets-v1.5.0.zip`)

### Models — **required** (sim + real)

`assets/models/g1/beyondmimic/`:

| ONNX | Configs |
|------|---------|
| `bts_2_0_tracking_2`, `bts_dynamite_tracking`, `easy_sample`, `Swim_tracking`, `thriller`, `salsa_tracking`, `gdance`, `Salsa_4`, `thriller_locked_waist` | `g1_switch_beyondmimic`, shazam |
| `g1_29dof_gesti`, `g1_29dof_gesti_10k`, `g1_29dof_67_10k`, `g1_29dof_67_safe`, `g1_29dof_gesti_safe`, `Violin` | gesti / real robot |

`assets/models/g1/unitree/`: `policy_lstm_1.pt`, `policy_wo_gait.pt` — **sim only**

### MuJoCo — **sim only**

`assets/robots/g1/` — XML + `meshes/*.STL`

### Audio — **music demos**

- `mp3_songs/*.mp3` (7 songs)
- `shazam/index.pkl`, `clap_index.json`, `clap_index.npy` (optional if rebuilt)
- `demo/real_try_trimmed_minus2s.mp3` → extracted to repo root

---

## Maintainer: publish a new release

```bash
# 1. Build index if needed
python shazam/run_experiment.py --build-index --songs-dir mp3_songs --index-path shazam/index.pkl
python shazam/run_experiment.py --build-clap-index --songs-dir mp3_songs --clap-index shazam/clap_index.json

# 2. Build zip (output: release/dist/ — gitignored)
./release/package_assets.sh

# 3. Create GitHub Release v1.5.0 and attach release/dist/robojudo-assets-v1.5.0.zip
gh release create v1.5.0 release/dist/robojudo-assets-v1.5.0.zip --title "Assets v1.5.0"
```

Update [assets_urls.yaml](assets_urls.yaml) if the tag or repo changes.

---

## Not in the asset zip

- Python packages → `pip install -e .`
- `unitree_sdk2` + `unitree_cpp` → real robot only
- HuggingFace CLAP weights → auto-download on first `--clap-fallback`
- Extra ONNX (`Dance_wose`, `Waltz`, …) → not used by release configs
- `.mp4` video — project uses `.mp3` mashups only
