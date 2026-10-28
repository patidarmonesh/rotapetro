# Execution notes

## Result verification only

Run `python tools/reproduce_report_metrics.py` from this repository. Only the dependencies in `requirements-report.txt` are needed. No GPU or raw data is required.

## Existing frozen V1 experiment

Restore the matching `data/interim` aligned stacks, maps and segmentation, plus V1 processed tables, frozen splits/hash and teacher prerequisites. Review paths in `configs/default.yaml`. Preserve original backups and do not mix outputs from different region-mask versions.

On a configured Linux GPU environment, the reported workflow is:

```bash
set -euo pipefail
bash scripts/run_pipeline.sh --v1 --config configs/default.yaml
python scripts/08_embed.py --config configs/default.yaml --device cuda
python scripts/v1_train.py --config configs/default.yaml --device cuda
python scripts/10_evaluate.py --config configs/default.yaml
```

`--v1` intentionally skips Phases 0–3. The source tests and feasibility gate are prerequisites, not evidence that raw data or valid masks automatically exist. If the gate fails, inspect the report and stop before training rather than weakening it merely to make execution proceed.

## Fresh preprocessing experiment

Restore the lawful raw dataset and use new experiment directories. Inspect the individual entry points `00_clean.py`, `01_align.py`, `02_maps.py`, `03_segment.py` and their configuration before running. Rebuilding masks changes region IDs; the corresponding feature tables, target tables, folds and embeddings must then be rebuilt consistently. Do not run an overwrite-forcing preprocessing command into a frozen historical result directory.

## Environment and provenance

- Install a PyTorch/CUDA build compatible with the actual GPU; do not assume the archived machine's versions fit another device.
- Application requirements are lower bounds, not an exact lock. Save the resolved environment after installation.
- Save the source commit, complete resolved YAML, input/target/split hashes and immutable encoder revision for each future run.
- Several transformer sizes/dropout choices are fixed in the current implementation rather than controlled by similarly named YAML fields.
- `configs/v1_reg.yaml` is not supplied as an exact historical file. The report lists the known regularised changes.
- Full 72-position unaveraged support is not implemented by these notes. It requires coordinated changes to preprocessing, indexing, cache and model evaluation.
