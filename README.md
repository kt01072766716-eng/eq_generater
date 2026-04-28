# Training data pipeline template

This template turns 5 standalone scripts into a reproducible **run-based** pipeline:

1) Generate target spectra (xlsx)
2) Split xlsx into per-spectrum txt files
3) Generate matched artificial ground motions for each target spectrum
4) Compute PSA for generated motions
5) Collect PSA outputs into ML-ready (X, y)

## Quick start

```bash
pip install -r requirements.txt
python -m src.pipeline --config configs/default.yaml --run-id 2025-12-29_1430
```

Outputs will be written to:

```
runs/<run-id>/
```

## Notes
- This template keeps your original scripts under `src/legacy/` untouched.
- New code under `src/steps/` re-implements the same logic but with **path arguments** and a shared config.
- python -m src.pipeline --config configs/default.yaml --run-id val --nproc 8
- python run_debug_range.py --config configs/default.yaml --run-id val12_yb_4p5 --start-step 4 --end-step 4