# RotaPetro V1 Evaluation Report

## 1. Primary Endpoint: Specimen-Balanced Macro-F1 on Singletons

| Arm | Macro-F1 (mean ± std) | Accuracy | Regions | Specimens | Per-Seed F1 |
|---|---:|---:|---:|---:|---|
| static_k1 | 0.4935 ± 0.0131 | 0.5368 | 581 | 77 | [0.4841, 0.4944, 0.4795, 0.4921, 0.5174] |
| unordered_k12 | 0.4724 ± 0.0106 | 0.5182 | 581 | 77 | [0.4855, 0.4842, 0.4696, 0.4613, 0.4614] |
| angle_aware_k12 | 0.4886 ± 0.0426 | 0.5347 | 581 | 77 | [0.5134, 0.5131, 0.4490, 0.5397, 0.4279] |

## 2. Paired Bootstrap Comparisons (5,000 replicates)

| Comparison | Point Diff | 95% Bootstrap CI | p-value (approx) |
|---|---:|---:|---:|
| primary_comparison | -0.0049 | [-0.0219, 0.0121] | 0.7118 |
| unordered_vs_static | -0.0211 | [-0.0393, -0.0031] | 0.9896 |
| angle_aware_vs_unordered | +0.0162 | [0.0003, 0.0344] | 0.0224 |

## 3. Partial-Set Compatibility & Probability Mass

| Arm | Top-1 Compatibility | Probability Mass in Set | Regions | Specimens |
|---|---:|---:|---:|---:|
| static_k1 | 0.7814 | 0.6944 | 323.0 | 29.0 |
| unordered_k12 | 0.7808 | 0.6833 | 323.0 | 29.0 |
| angle_aware_k12 | 0.7753 | 0.6920 | 323.0 | 29.0 |

## 4. Static Arm Angle Sensitivity

- **Mean Macro-F1 across angles**: 0.4799 ± 0.0089
- **Range [min, max]**: [0.4641, 0.4935]
- **Preregistered Primary Test Angle (Index 1 = 5°)**: 0.4935

| Angle Index | Angle (deg) | Macro-F1 |
|---:|---:|---:|
| 0 | 0.0° | 0.4883 |
| 1 | 5.0° | 0.4935 |
| 4 | 20.0° | 0.4831 |
| 7 | 35.0° | 0.4733 |
| 10 | 50.0° | 0.4684 |
| 13 | 65.0° | 0.4641 |
| 16 | 80.0° | 0.4828 |
| 19 | 95.0° | 0.4907 |
| 22 | 110.0° | 0.4853 |
| 25 | 125.0° | 0.4762 |
| 28 | 140.0° | 0.4762 |
| 31 | 155.0° | 0.4697 |
| 34 | 170.0° | 0.4872 |

## 5. Performance Split by Metadata Dependency

| Arm | Stratum | Macro-F1 | Regions | Specimens | n regions / n specimens |
|---|---|---:|---:|---:|---:|
| static_k1 | metadata_dependent | 0.4519 | 537 | 71 | 537 / 71 |
| static_k1 | optics_only_reproducible | unavailable (n=44) | 44 | 20 | 44 / 20 |
| unordered_k12 | metadata_dependent | 0.4378 | 537 | 71 | 537 / 71 |
| unordered_k12 | optics_only_reproducible | unavailable (n=44) | 44 | 20 | 44 / 20 |
| angle_aware_k12 | metadata_dependent | 0.4512 | 537 | 71 | 537 / 71 |
| angle_aware_k12 | optics_only_reproducible | unavailable (n=44) | 44 | 20 | 44 / 20 |

## 6. Static Strata Performance & Comparisons

| Stratum | Arm | Macro-F1 | Regions | Specimens | n regions / n specimens | Paired Bootstrap Diff vs static_k1 (95% CI, p) |
|---|---|---:|---:|---:|---:|---|
| reproduced | static_k1 | 0.3569 | 77 | 25 | 77 / 25 | - |
| reproduced | unordered_k12 | 0.3429 | 77 | 25 | 77 / 25 | - |
| reproduced | angle_aware_k12 | 0.3582 | 77 | 25 | 77 / 25 | +0.0013 [-0.0311, 0.0320] (p=0.4316) |
| rotation_dependent | static_k1 | 0.4323 | 501 | 69 | 501 / 69 | - |
| rotation_dependent | unordered_k12 | 0.4197 | 501 | 69 | 501 / 69 | - |
| rotation_dependent | angle_aware_k12 | 0.4313 | 501 | 69 | 501 / 69 | -0.0011 [-0.0173, 0.0155] (p=0.5468) |
