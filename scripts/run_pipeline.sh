#!/usr/bin/env bash
# ==============================================================================
# RotaPetro Pipeline: Phase 0 → 7 (with V1 Support and Quality Gates)
#
# Usage:
#   bash scripts/run_pipeline.sh [--v1] [--config configs/default.yaml]
#
# Safety Invariants Enforced:
#   1. Strict Shell Safety:
#      - Script runs with 'set -euo pipefail' and explicit failure traps to halt
#        immediately on any non-zero exit code.
#   2. Frozen Segmentation Preservation (V1 Mode):
#      - In V1 mode, Phases 0-3 (clean, align, maps, segment) are skipped.
#      - Rebuilding segmentation would regenerate region IDs and contours,
#        invalidating all frozen splits and per-fold targets.
#   3. Feature Extraction Single Source of Truth (Phase 4):
#      - Phase 4 extracts grain property cards and computes V1_STATUS_COLUMNS
#        (optic_class_status, pleo_status, absorption_status, twin_status, extinction_status, cleavage_status)
#        and pixel eligibility (eligible_v1).
#      - Output data/processed/v1/regions.parquet is required by Phase 6 & Phase 7.
#   4. Knowledge Base Validation (Phase 5):
#      - Validates that mineral taxa and group definitions load without error
#        before labelling rules execute.
#   5. Per-Fold Teacher Isolation & Quarantined Terms (Phase 6 V1):
#      - Autolabelling runs strictly per-fold (0..4) without cross-fold leakage.
#      - Quarantined terms (birefringence, extinction_type, relief, cleavage, absorption_axis) are disabled
#        from deciding terms.
#   6. Quality & Feasibility Gate Halt Condition (Phase 6.1 V1 Gate):
#      - scripts/v1_label_feasibility.py evaluates the 5-family gate and enforces:
#          * V1_STATUS_COLUMNS exist in regions.parquet
#          * Exactly 5 fold targets exist with matching row count
#          * 'fine_aggregate' is NEVER mapped to singleton
#          * deciding_terms contains 0 birefringence and 0 extinction_type
#          * >= 5 singleton training specimens per class in all 5 folds
#          * Opaque material lightness check (ppl_L and xpl_rel_L)
#      - CRITICAL: If scripts/v1_label_feasibility.py fails (exit code != 0) or
#        overall_gate_pass is False, the pipeline MUST halt immediately.
#        Phase 7 dataset assembly MUST NEVER run on failing labels.
#   7. Dataset Assembly & Frozen Split Integrity (Phase 7 V1):
#      - Merges eligible regions with segment geometry, validates bboxes, and
#        verifies that splits/folds.json strictly matches FOLDS_SHA256.txt.
# ==============================================================================

set -euo pipefail

V1_MODE=false
CONFIG="configs/default.yaml"

for arg in "$@"; do
  case "$arg" in
    --v1)
      V1_MODE=true
      ;;
    --config=*)
      CONFIG="${arg#*=}"
      ;;
    *)
      if [[ "$arg" != --* ]] && [[ -f "$arg" ]]; then
        CONFIG="$arg"
      fi
      ;;
  esac
done

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

log()  { echo -e "\n\033[1;36m══════ $1 ══════\033[0m"; }
ok()   { echo -e "\033[1;32m  ✓ $1\033[0m"; }
fail() { echo -e "\n\033[1;31m  ✗ FATAL: $1\033[0m\n" >&2; exit 1; }

# ── Phase 0 to 3: Raw Image Ingestion & Segmentation ─────────────────────────
# In V1 mode, skip phases 0-3 to prevent rebuilding segmentation and altering region IDs,
# which would invalidate all frozen V1 targets and split assignments (Safety Invariant 2).
if [ "$V1_MODE" = false ]; then
  log "Phase 0 — Data Cleaning & Deduplication"
  python scripts/00_clean.py --config "$CONFIG" || fail "Phase 0 (00_clean.py) failed"
  ok "specimens_clean.csv and sites.csv written"

  log "Phase 1 — Alignment"
  python scripts/01_align.py --config "$CONFIG" --force || fail "Phase 1 (01_align.py) failed"
  ok "aligned/ stacks written"

  log "Phase 2 — Property Maps"
  python scripts/02_maps.py --config "$CONFIG" --force || fail "Phase 2 (02_maps.py) failed"
  ok "maps/ written"

  log "Phase 3 — Segmentation"
  python scripts/03_segment.py --config "$CONFIG" --force || fail "Phase 3 (03_segment.py) failed"
  ok "segments/ written"
else
  log "V1 Mode Active: Preserving frozen segmentation (Phases 0-3 skipped)"
  ok "Retaining established site segments and stable region IDs"
fi

# ── Phase 4: Feature Extraction ──────────────────────────────────────────────
# Safety Invariant 3: Extract grain property cards and populate V1 status columns.
# Output regions.parquet is the single source of truth for downstream phases.
log "Phase 4 — Feature Extraction"
python scripts/04_features.py --config "$CONFIG" || fail "Phase 4 (04_features.py) failed"
ok "regions.parquet written and validated"

# ── Phase 5: Knowledge Base Validation ───────────────────────────────────────
# Safety Invariant 4: Validate knowledge base sanity before labelling execution.
log "Phase 5 — Knowledge Base (validation only)"
python -c "
import sys; sys.path.insert(0,'.')
from src.rotapetro.knowledge import KnowledgeBase
kb = KnowledgeBase('knowledge')
n = len(kb.all_minerals())
assert n > 0, 'No minerals loaded'
print(f'  Knowledge base OK: {n} minerals, {len(kb.all_groups())} groups')
" || fail "Phase 5 Knowledge base validation failed"
ok "knowledge validated"

# ── Phases 6 & 7: Labelling, Feasibility Gate & Dataset Assembly ─────────────
if [ "$V1_MODE" = true ]; then
  # ── Phase 6 (V1): Per-Fold Teachers ────────────────────────────────────────
  # Safety Invariant 5: Generate per-fold targets across frozen folds 0..4.
  # Quarantined terms (birefringence, extinction_type, relief, cleavage, absorption_axis) are strictly excluded.
  log "Phase 6 (V1) — Automatic Labels & Per-Fold Teachers"
  python scripts/06_autolabel.py --config "$CONFIG" --v1 || fail "Phase 6 V1 (06_autolabel.py --v1) failed"
  ok "V1 per-fold targets generated under teachers/fold_*/targets.parquet"

  # ── Phase 6.1 (V1 Gate): Label Feasibility & Quality Gate ──────────────────
  # Safety Invariant 6: Feasibility Gate Halt Condition.
  # If scripts/v1_label_feasibility.py fails (exit code != 0), the script immediately
  # stops. Phase 7 (Dataset Assembly) MUST NOT be executed on failing label sets.
  log "Phase 6.1 (V1 Gate) — Label Feasibility & Quality Evaluation"
  if ! python scripts/v1_label_feasibility.py --config "$CONFIG"; then
    fail "scripts/v1_label_feasibility.py failed (exit code != 0). Halting pipeline immediately before Phase 7."
  fi

  # Explicit secondary check: verify overall_gate_pass in provenance
  python -c "
import json, sys
from pathlib import Path
from src.rotapetro.io import load_config, resolve_v1_paths
cfg = load_config('$CONFIG')
v1_paths = resolve_v1_paths(cfg)
prov_path = Path(v1_paths['processed']) / 'provenance_feasibility.json'
if not prov_path.is_file():
    print(f'ERROR: Provenance file missing at {prov_path}', file=sys.stderr)
    sys.exit(1)
prov = json.loads(prov_path.read_text(encoding='utf-8'))
if not prov.get('overall_gate_pass', False):
    print('ERROR: Feasibility gate overall_gate_pass is False!', file=sys.stderr)
    sys.exit(1)
" || fail "Feasibility gate overall_gate_pass check failed. Halting pipeline before Phase 7."
  ok "V1 per-fold targets written and label feasibility gate passed"

  # ── Phase 7 (V1): Dataset Assembly ─────────────────────────────────────────
  # Safety Invariant 7: Assemble final training dataset and verify frozen folds.
  # Strictly chained after Phase 6 and the feasibility gate have succeeded.
  log "Phase 7 (V1) — Dataset Assembly (Frozen Splits)"
  python scripts/07_build_dataset.py --config "$CONFIG" --v1 || fail "Phase 7 V1 (07_build_dataset.py --v1) failed"
  ok "data/processed/v1/samples.parquet written (folds verified against FOLDS_SHA256.txt)"

  python scripts/visualize_labels_v1.py --config "$CONFIG" --fold 0 || echo "WARN: visual review failed"

  log "ALL DONE — V1 Preprocessing complete (Phases 4 → 6 → 7 chained & verified)"
  echo ""
  echo "  Next steps for V1:"
  echo "    python scripts/08_embed.py --config $CONFIG"
  echo "    python scripts/v1_train.py --config $CONFIG"
  echo "    python scripts/10_evaluate.py --config $CONFIG"
else
  # ── Legacy Pipeline Chaining ───────────────────────────────────────────────
  log "Phase 6 (Legacy) — Automatic Labels"
  python scripts/06_autolabel.py --config "$CONFIG" --force || fail "Phase 6 Legacy failed"
  ok "labels.parquet + classes.yaml written"

  log "Phase 7 (Legacy) — Dataset Assembly & Splits"
  python scripts/07_build_dataset.py --config "$CONFIG" || fail "Phase 7 Legacy failed"
  ok "samples.parquet + folds.json + class_counts.json written"

  log "ALL DONE — Legacy Pipeline complete (Phase 0 → 7)"
fi
