# RotaPetro V1 Evaluation Report

## 1. Primary Endpoint: Specimen-Balanced Macro-F1 on Singletons

| Arm | Macro-F1 (mean ± std) | Accuracy | Regions | Specimens | Per-Seed F1 |
|---|---:|---:|---:|---:|---|
| static_k1 | 0.5309 ± 0.0114 | 0.5406 | 581 | 77 | [0.5468, 0.5207, 0.5251] |
| unordered_k12 | 0.5073 ± 0.0362 | 0.5213 | 581 | 77 | [0.5309, 0.4562, 0.5349] |
| angle_aware_k12 | 0.5308 ± 0.0647 | 0.5391 | 581 | 77 | [0.5049, 0.4678, 0.6198] |

## 2. Paired Bootstrap Comparisons (5,000 replicates)

| Comparison | Point Diff | 95% Bootstrap CI | p-value (approx) |
|---|---:|---:|---:|
| primary_comparison | -0.0000 | [-0.0449, 0.0461] | 0.4982 |
| unordered_vs_static | -0.0235 | [-0.0692, 0.0206] | 0.8488 |
| angle_aware_vs_unordered | +0.0235 | [-0.0270, 0.0773] | 0.1836 |

## 3. Partial-Set Compatibility & Probability Mass

| Arm | Top-1 Compatibility | Probability Mass in Set | Regions | Specimens |
|---|---:|---:|---:|---:|
| static_k1 | 0.7600 | 0.7016 | 323.0 | 29.0 |
| unordered_k12 | 0.7541 | 0.6922 | 323.0 | 29.0 |
| angle_aware_k12 | 0.7552 | 0.6881 | 323.0 | 29.0 |

## 4. Static Arm Angle Sensitivity

- **Mean Macro-F1 across angles**: 0.5056 ± 0.0104
- **Range [min, max]**: [0.4911, 0.5309]
- **Preregistered Primary Test Angle (Index 1 = 5°)**: 0.5309

| Angle Index | Angle (deg) | Macro-F1 |
|---:|---:|---:|
| 0 | 0.0° | 0.5051 |
| 1 | 5.0° | 0.5309 |
| 4 | 20.0° | 0.5116 |
| 7 | 35.0° | 0.5087 |
| 10 | 50.0° | 0.4983 |
| 13 | 65.0° | 0.4911 |
| 16 | 80.0° | 0.5042 |
| 19 | 95.0° | 0.5090 |
| 22 | 110.0° | 0.5188 |
| 25 | 125.0° | 0.4965 |
| 28 | 140.0° | 0.5068 |
| 31 | 155.0° | 0.4940 |
| 34 | 170.0° | 0.4979 |

## 5. Performance Split by Metadata Dependency

| Arm | Stratum | Macro-F1 | Regions | Specimens | n regions / n specimens |
|---|---|---:|---:|---:|---:|
| static_k1 | metadata_dependent | 0.4376 | 537 | 71 | 537 / 71 |
| static_k1 | optics_only_reproducible | unavailable (n=44) | 44 | 20 | 44 / 20 |
| unordered_k12 | metadata_dependent | 0.4333 | 537 | 71 | 537 / 71 |
| unordered_k12 | optics_only_reproducible | unavailable (n=44) | 44 | 20 | 44 / 20 |
| angle_aware_k12 | metadata_dependent | 0.4506 | 537 | 71 | 537 / 71 |
| angle_aware_k12 | optics_only_reproducible | unavailable (n=44) | 44 | 20 | 44 / 20 |

## 6. Static Strata Performance & Comparisons

| Stratum | Arm | Macro-F1 | Regions | Specimens | n regions / n specimens | Paired Bootstrap Diff vs static_k1 (95% CI, p) |
|---|---|---:|---:|---:|---:|---|
| reproduced | static_k1 | 0.4020 | 77 | 25 | 77 / 25 | - |
| reproduced | unordered_k12 | 0.3838 | 77 | 25 | 77 / 25 | - |
| reproduced | angle_aware_k12 | 0.4137 | 77 | 25 | 77 / 25 | +0.0117 [-0.0493, 0.0620] (p=0.3974) |
| rotation_dependent | static_k1 | 0.4139 | 501 | 69 | 501 / 69 | - |
| rotation_dependent | unordered_k12 | 0.4155 | 501 | 69 | 501 / 69 | - |
| rotation_dependent | angle_aware_k12 | 0.4269 | 501 | 69 | 501 / 69 | +0.0130 [-0.0237, 0.0507] (p=0.2364) |
