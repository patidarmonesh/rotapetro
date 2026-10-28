# Report evidence verification

Checked on 2026-10-28 with `tools/reproduce_report_metrics.py`:

- All 120 full-fit prediction files were read (45 initial, 75 regularised).
- Probability entries were finite, nonnegative and summed to one within tolerance.
- Saved top-class predictions matched probability argmax.
- Each seed contained all five held-out folds with no repeated out-of-fold region.
- Each specimen appeared in only one held-out fold.
- Region identities, specimen IDs, fold assignments, states and allowed targets matched across arms for the same seed.
- Each pooled singleton evaluation contained 581 regions from 77 specimens.
- Recomputed mean F1 agreed with the archived summaries to their displayed precision.

| Setting | Static | Unordered | Angle-aware | Angle-aware minus static |
|---|---:|---:|---:|---:|
| Initial | 0.530866 | 0.507335 | 0.530824 | -0.00004142 |
| Regularised | 0.493522 | 0.472406 | 0.488589 | -0.00493314 |

This audit establishes consistency of saved prediction arithmetic and joins. It does not independently verify mineral identities, reproduce the historical training environment or prove absence of every possible dataset bias. No GPU training was run while preparing this report.

## Document verification

- The final report has 35 pages, 37 numbered mathematical expressions and 13 numbered figures, plus the cover process diagram.
- LaTeX compilation succeeded with Tectonic and with the desktop editor's compiler.
- No unresolved references/citations or overfull text boxes remained in the final compiler log.
- Every PDF page was rendered and visually reviewed; selected equation, chart, table and example pages were additionally inspected at larger size.
- Copied source/evidence file hashes were checked against the manifest. Application source was copied without edits.
