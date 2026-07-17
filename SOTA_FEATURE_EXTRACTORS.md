# SOTA Feature Extractors for Linear Probing

This branch wires the linear-probing script to the SOTA encoders used in the paper table:

| Feature mode | Source | Notes |
| --- | --- | --- |
| `surgvlp` | `CAMMA-public/SurgVLP` | Uses the released SurgVLP visual encoder. |
| `hecvl` | `CAMMA-public/SurgVLP` | Uses the released HecVL visual encoder. |
| `surgclip_beta` | `aperezr20/SurgLaVi` / PyPI `surgclip` | Uses the SurgCLIP-beta visual encoder and projection head. |
| `surgalign` or `vlp` | local code | Uses this repository's `VLP` checkpoint. |

The official SurgVLP inference package is vendored under `third_party/SurgVLP`.
The SurgCLIP-beta package was already present under `surgclip/` and matches the official
`src/surgclip` tree from `aperezr20/SurgLaVi` aside from local cache files.

## Usage

Run one model:

```bash
FEATURE_MODE=surgvlp DATASETS=cholec80_phase SHOT_MODE=video SHOTS=0.1,0.5,1.0 bash ./run_linear_probe.sh
FEATURE_MODE=hecvl DATASETS=cholec80_phase SHOT_MODE=video SHOTS=0.1,0.5,1.0 bash ./run_linear_probe.sh
FEATURE_MODE=surgclip_beta DATASETS=cholec80_phase SHOT_MODE=video SHOTS=0.1,0.5,1.0 bash ./run_linear_probe.sh
FEATURE_MODE=surgalign CKPT=/path/to/vlp_epoch.pt DATASETS=cholec80_phase bash ./run_linear_probe.sh
```

Run all table baselines:

```bash
SURGALIGN_CKPT=/path/to/vlp_epoch.pt bash ./run_linear_probe_sota_models.sh
```

Run the SurgLaVi-aligned linear probing protocol:

```bash
SURGALIGN_CKPT=/path/to/vlp_epoch.pt bash ./run_linear_probe_surglavi_protocol.sh
```

This uses `cholec80_phase,autolaparo_phase,grasp_phase,grasp_step`, five seeds,
CLS shots `1,2,4,8,16`, and video shots `0.1,0.5,1.0`. To match SurgLaVi's
32-frame temporal context while respecting SurgAlign's 8-frame temporal module,
the script defaults to `CONTEXT_NUM_FRAMES=32`, `NUM_FRAMES=8`, and
`CONTEXT_STRIDE=8`: each 32-frame context is encoded as four 8-frame chunks and
the chunk embeddings are mean-pooled before training the linear head. The
classifier batch size is set to 256. `ENCODE_BATCH_SIZE` defaults to 16 only to
reduce memory pressure during frozen-feature extraction.

For offline runs, pass local checkpoints:

```bash
SURGVLP_CKPT=/path/to/SurgVLP.pth \
HECVL_CKPT=/path/to/HecVL.pth \
SURGCLIP_BETA_CKPT=/path/to/surgclip_beta.pth \
SURGALIGN_CKPT=/path/to/vlp_epoch.pt \
bash ./run_linear_probe_sota_models.sh
```

If an external checkpoint variable is empty, the adapter falls back to the official package downloader/cache.

## License

`third_party/SurgVLP` follows the upstream non-commercial research license described in its README
(CC BY-NC-SA 4.0 terms for code/models). The local `surgclip/` package metadata declares MIT for
the PyPI package, while the SurgLaVi dataset card uses CC BY-NC-SA terms for the dataset.
