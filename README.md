# Quantization Experiments and Inference Benchmarking

Reproducible scripts and utilities to benchmark inference performance and evaluate precision and quantization choices
(FP32, mixed precision, and INT8 PTQ/QAT) on CPU and GPU across medical imaging datasets.

## Status

Submitted to CHIL (Conference on Health, Inference, and Learning) 2026.

## Author

Okan Bilge Ozdemir

## What is included

- **Runnable scripts** in `scripts/`
  - `benchmark_inference_fair_v2.py`: inference-only latency and throughput benchmarking (CPU and GPU where available)
  - `run_precision_experiment.py`: runs precision experiments from a YAML config
  - `run_qat_experiment.py`: runs quantization-aware training (QAT) experiments from a YAML config
  - `generate_all_figures.py`: generates figures from experiment outputs
- **Reusable helpers** in `utils/` and `src/quant_bench/utils/`
  - data loading, metrics, model factory, training and evaluation, quantization helpers
- **Configuration** in `configs/config.yaml`
- **Output structure**
  - `runs/` for experiment outputs
  - `cv_splits/` for saved cross-validation splits (optional)

## Repository layout

```
configs/                 YAML configs
scripts/                 runnable entry points
utils/                   helper modules (kept for compatibility with existing imports)
src/quant_bench/utils/    installable package copy of helpers
data/                    place datasets here (not tracked)
runs/                    outputs (not tracked)
cv_splits/               saved CV splits (not tracked)
docs/                    extra notes
examples/                example commands
```

## Setup

### 1) Create and activate an environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Prepare datasets

By default, `configs/config.yaml` expects each dataset under:

- `./data/chestxray`
- `./data/brainmri`
- `./data/skincancer`

Update the `data_root` fields in `configs/config.yaml` if your datasets live elsewhere.

### 3) Run

Run a precision experiment:

```bash
python scripts/run_precision_experiment.py --config configs/config.yaml
```

Run a QAT experiment:

```bash
python scripts/run_qat_experiment.py --config configs/config.yaml
```

Run inference benchmarking:

```bash
python scripts/benchmark_inference_fair_v2.py --config configs/config.yaml
```

Outputs are written under `./runs` by default. To change the output root:

```bash
export EXPERIMENT_ROOT=/path/to/outputs
```

## Citation

A citation entry will be added after acceptance and publication. For now, please cite this repository as:

Ozdemir, Okan Bilge. Quantization Experiments and Inference Benchmarking (code repository). Submitted to CHIL.

## Notes

- PyTorch eager-mode INT8 quantized models run on CPU only (not CUDA) in standard eager mode.
- If you need GPU INT8, consider TensorRT or ONNX Runtime CUDA INT8.

## License

See `LICENSE`.
