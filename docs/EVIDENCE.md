# Evidence map

| Claim | Source | Qualification |
|---|---|---|
| Initial/regularised F1 and bootstrap interval | `results/*/evaluation_summary.json` | Saved historical evaluation against weak targets |
| Region-wise predicted probabilities | `results/*/<arm>/<fold>/<seed>/predictions.parquet` | Test predictions; not verified mineral labels |
| Learning curves and early checkpoint selection | Matching `train_log.csv`, `train_summary.json` | Epoch indices in audit tables may be zero-based |
| Training support gate | `evidence/phase7/label_feasibility_report.md` | Historical Phase 7 evidence; do not substitute current regenerated labels |
| Current algorithm | `src/`, `scripts/`, `configs/`, `knowledge/` | Current explanatory snapshot; exact historical GPU checkout not established |
| Preprocessing inventory | `evidence/local_preprocessing/` | Local outputs; not all source arrays included |
| Granite example | `evidence/granite_example.json`, report images | Local masks/crops with saved fold 3, seed 0 probabilities; GPU cache absent |
| Current table columns | `evidence/*_schema.csv` | Schemas, not historical training data |

The original result archives were `rotapetro_v1_results.tar.gz` and `rotapetro_v1_reg_results.tar.gz`. The companion package retains compact prediction/evaluation files from their extracted directories. It excludes smoke runs, model checkpoints, full embedding caches and raw arrays. The copying manifest records original relative paths and content hashes. Some files described in the report dictionary, such as training provenance records, were not present at every archived run location; the dictionary describes their role, not a guarantee that every such artefact survived packaging.

In `source_manifest.json`, `sha256` describes the copied original bytes; `packaged_sha256` describes the Git-normalised packaged bytes. Git normalises text line endings to LF for portability. This can change a byte hash without changing source logic or numerical values.

Regularised settings supplied with the run: learning rate 1e-4, weight decay 0.2, maximum 150 epochs, patience 30, seeds 0–4, result directory `outputs/v1_reg`. A complete original `configs/v1_reg.yaml` was not supplied. Do not label a reconstructed file as the exact historical configuration.

The primary report intervals are the archived stratified paired specimen-bootstrap intervals. The separate historical audit includes slightly different independent resampling analyses; those are diagnostic and are not silently substituted for the saved primary interval.

Report preparation does not alter the application source or launch training. The pipeline source in this package is copied for review. A dedicated evidence verifier checks arithmetic independently.
