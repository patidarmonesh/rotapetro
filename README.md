# RotaPetro — rotation-aware mineral-family classification

Research code, technical report and compact prediction evidence for a comparison of single-view, unordered multi-view and angle-aware classifiers on rock thin sections.

**Start with [the technical report](report/RotaPetro_Technical_Report.pdf).** Its editable source is [report/main.tex](report/main.tex). The report explains the research questions, each phase, mathematics, visual outputs, completed experiments and limitations. No presentation is included.

## Main finding

| Model | Initial specimen-weighted macro-F1 | Regularised macro-F1 |
|---|---:|---:|
| Static, one PPL/XPL pair | 0.5309 | 0.4935 |
| Unordered, 12 pairs | 0.5073 | 0.4724 |
| Angle-aware, 12 pairs | 0.5308 | 0.4886 |

The primary angle-aware-minus-static intervals include zero: **[-0.0449, 0.0461]** initially and **[-0.0219, 0.0121]** after regularisation. The completed experiment did not demonstrate a rotation advantage. These are agreements with automatic weak labels on 581 singleton regions from 77 specimens, not verified mineral accuracy.

The current workflow takes 72 raw positions (144 photographs), aligns and averages opposite-angle pairs into 36 retained positions, encodes all 36, and jointly classifies 12 PPL/XPL pairs per multi-view decision. A full unaveraged 72-position experiment is proposed, not completed.

## Contents

- `src/`, `scripts/`, `tests/`, `configs/`, `knowledge/`: current reviewed source snapshot.
- `report/`: LaTeX report, equations, figures and compiled PDF.
- `results/baseline/`, `results/regularized/`: historical saved predictions and fit/evaluation summaries for 45 and 75 fits.
- `evidence/`: archived feasibility evidence, local preprocessing summaries, schemas, hashes and provenance.
- `tools/reproduce_report_metrics.py`: independent point-estimate verification from saved predictions.
- `docs/`: execution notes, evidence boundaries and source attribution.

Raw photographs, full preprocessing arrays, embedding caches, checkpoints and Python environments are not included. The original web downloader is absent from the reviewed project. This is a source-and-evidence repository, not a self-contained raw-data distribution.

## Recompute the primary point estimates (CPU)

From the repository root:

```bash
python -m venv .venv
# Activate with: source .venv/bin/activate (Linux/macOS)
# Windows PowerShell: .venv/Scripts/Activate.ps1
python -m pip install -r requirements-report.txt
python tools/reproduce_report_metrics.py
```

The verifier pools all five held-out folds for each seed, checks matching region/target identities across arms, builds an inverse-specimen-weighted confusion matrix, and compares mean macro-F1 with archived evaluation summaries. It does not retrain models or silently regenerate targets. Historical bootstrap intervals are retained as reported; this utility verifies their underlying point estimates only.

## Build the report

With [Tectonic](https://tectonic-typesetting.github.io/) installed:

```bash
cd report
tectonic main.tex
```

This writes `main.pdf`; the packaged final copy is named `RotaPetro_Technical_Report.pdf`. Alternatively use `latexmk -pdf main.tex` with a suitable TeX distribution. Figures are included, so no GPU, raw imagery or Python execution is needed to compile the document. A first Tectonic build may download its TeX resource bundle.

## Running the research pipeline

Read [docs/RUNNING.md](docs/RUNNING.md) before using the source. In particular, `--v1` skips Phases 0–3 and requires frozen preprocessing/splits already available. Do not rebuild segmentation over historical experiment outputs: that can invalidate region identities. Install GPU-compatible PyTorch for the target machine; the broad application requirements are not a historical environment lock.

## Evidence boundaries

The snapshot was taken on 2026-10-28. `evidence/snapshot.json` records the original local source commit. The exact historical GPU checkout and complete regularised-run YAML were not preserved together. Current source explains the pipeline but is not certified as the identical historical checkout. Result figures use historical predictions; visual crops are local reconstructions. See [docs/EVIDENCE.md](docs/EVIDENCE.md).

## GitHub

This folder is a separate local Git repository with an initial packaging commit on `codex/technical-report`. Create an empty destination repository and add its URL as the remote before pushing. No public repository has been created or pushed by this packaging step. The ZIP contains the committed files without the `.git` history; use this local folder if you want the prepared Git repository.

## Attribution and licensing

Microscope images in the report derive from The Open University Virtual Microscope collection. [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) preserves the source and retained licence notice. No blanket licence is imposed over third-party images or the entire project. The reviewed source tree did not contain a standalone licence grant; see [LICENSE_NOTES.md](LICENSE_NOTES.md) before describing this repository as open-source or granting reuse permissions.
