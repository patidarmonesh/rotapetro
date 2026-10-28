# RotaPetro — Label-free Mineral Identification from Polarised-Light Stage Rotation

**Project specification, v1.0 (27 Sep 2026)**
Team project, Department of Earth Sciences, IIT Kanpur. Supervisor: _name_. Maintainer: _name_.

This document is the single source of truth for the project. Anyone who follows it step by step should be able to rebuild the dataset, the model, the evaluation and the inference tool without guessing. Wherever a number is given (threshold, size, learning rate) it is a **starting default**; the "Tuning" notes say how to adjust it.

---

## Table of contents

0. [One-page summary](#0-one-page-summary)
1. [Goals, non-goals and hard constraints](#1-goals-non-goals-and-hard-constraints)
2. [How this differs from LITHOS, and what to reuse or remove from the LITHOS code](#2-how-this-differs-from-lithos-and-what-to-reuse-or-remove-from-the-lithos-code)
3. [Repository layout and file locations](#3-repository-layout-and-file-locations)
4. [Environment and hardware](#4-environment-and-hardware)
5. [Raw data (what the downloader produced)](#5-raw-data-what-the-downloader-produced)
6. [Phase 0 — Clean and audit the download](#6-phase-0--clean-and-audit-the-download)
7. [Phase 1 — Align the rotation frames](#7-phase-1--align-the-rotation-frames)
8. [Phase 2 — Optical property maps (per pixel)](#8-phase-2--optical-property-maps-per-pixel)
9. [Phase 3 — Automatic grain segmentation](#9-phase-3--automatic-grain-segmentation)
10. [Phase 4 — Grain property cards (features)](#10-phase-4--grain-property-cards-features)
11. [Phase 5 — Knowledge base (the "textbook" table)](#11-phase-5--knowledge-base-the-textbook-table)
12. [Phase 6 — Automatic labels (no human labelling)](#12-phase-6--automatic-labels-no-human-labelling)
13. [Phase 7 — Training dataset assembly and splits](#13-phase-7--training-dataset-assembly-and-splits)
14. [Phase 8 — The model (RotaPetro-Net)](#14-phase-8--the-model-rotapetro-net)
15. [Phase 9 — Training procedure](#15-phase-9--training-procedure)
16. [Phase 10 — Evaluation without human labels](#16-phase-10--evaluation-without-human-labels)
17. [Phase 11 — Inference: from a rotation video to a mineral report](#17-phase-11--inference-from-a-rotation-video-to-a-mineral-report)
18. [Expected results, success criteria and the paper](#18-expected-results-success-criteria-and-the-paper)
19. [Limitations and risks](#19-limitations-and-risks)
20. [Timeline and definition of done](#20-timeline-and-definition-of-done)
21. [Common mistakes to avoid](#21-common-mistakes-to-avoid)
22. [Appendices](#22-appendices)

---

## 0. One-page summary

**What we build:** a model that watches a thin-section field rotating under the microscope (plane-polarised light, PPL, and cross-polarised light, XPL) and, for every grain in that field, says which mineral it is. It also explains its answer in the language a petrologist uses: "strongly pleochroic brown, parallel extinction, high birefringence, one cleavage → biotite".

**Data:** the 115 thin sections of the Open University "UK Virtual Microscope" (UKVM) collection. Each has 1–4 rotation sites. A site is a circular field about 2 mm across, imaged as 72 PPL + 72 XPL frames at 5° steps over 360°. Each specimen also comes with a rock name and a list of the minerals present. There are **no grain-level labels** on the site.

**Key idea (no human labelling):** rotation lets us *measure* the optical properties of every grain:
- extinction position and angle
- isotropic, opaque or anisotropic
- pleochroism
- birefringence, read from the brightest interference colour
- twinning and undulose extinction
- cleavage

A knowledge base built from optical mineralogy textbooks turns these measurements into mineral labels. The choice is restricted to the minerals the OU page lists for that rock. Only confident labels are kept. A neural network then learns from these automatic labels and from the rock-level mineral lists.

**The one main model:** RotaPetro-Net.
- *Input:* the 36-angle PPL + XPL image sequence of one grain.
- *Parts:* a frozen DINOv2 image encoder, then a small angle-aware Transformer.
- *Outputs:* a mineral class, a mineral group, and predicted optical properties (for explanations).

**Headline experiment:** the same network given only the 0° frame (like LITHOS) versus the rotation sequence, plus how many angles are really needed (1, 2, 4, 6, 9, 18, 36).

**Inference:** a user records a short rotation video of one field in PPL and in XPL following our protocol. The pipeline returns:
- a grain map with the mineral, confidence and reasons for every grain;
- the minerals present in the field with their area %;
- an optional rock-type hint.

**Main limitations:**
- The dataset is small (115 specimens, a few thousand grains) and comes from one source and one microscope.
- Labels are automatic, so some are wrong.
- A few mineral pairs cannot be separated without optic sign / 2V or reflected light.

```
 OU download ─► Phase 0 clean ─► Phase 1 align ─► Phase 2 property maps ─► Phase 3 segment grains
                                                                                   │
            Phase 5 knowledge base ──────────────────────────► Phase 6 auto-labels ◄─ Phase 4 property cards
                                                                                   │
                  Phase 7 dataset + splits ─► Phase 8/9 RotaPetro-Net training ─► Phase 10 evaluation
                                                                                   │
                                                  Phase 11 inference on user rotation videos (report)
```

---

## 1. Goals, non-goals and hard constraints

### 1.1 Goals
- **G1.** A clean, documented rotation-series dataset derived from UKVM. It covers aligned PPL/XPL stacks, property maps, grain segmentations, grain property cards and automatic grain labels with confidence.
- **G2.** A rule-based, explainable mineral identifier (the "rule engine"). It needs no training and serves as both the label generator and a baseline.
- **G3.** One trained model, RotaPetro-Net, that identifies minerals from rotation sequences and outputs explanations.
- **G4.** A measured answer to the research question: how much does stage rotation improve mineral identification over a single 0° PPL+XPL view, and how many angles are needed?
- **G5.** An inference tool that turns a user's rotation video (recorded by our protocol) into a mineral report.
- **G6.** A paper plus an open release of code and derived labels. The release follows the OU licence (CC BY-NC-SA 2.0).

### 1.2 Non-goals (explicitly out of scope for v1)
- Whole-slide mineral maps from 0° images. There is no teacher–student transfer to whole slides; that is LITHOS-style work and possible future work.
- Rock naming from a single field as a hard output. It is only a "hint".
- Identifying opaque minerals (needs reflected light) or using optic sign / 2V (needs conoscopy).
- Training on or evaluating with human-drawn grain labels.

### 1.3 Hard constraints
- **C1. No human labelling of grains.** Humans may only inspect QC images to tune thresholds.
- **C2. Splits by specimen.** Every split is at the specimen level. Regions or sites of one specimen never appear in both train and test.
- **C3. Physics-consistent augmentation.** Any geometric augmentation must keep the angle between the polariser and the crystal physically valid (§15.4).
- **C4. No lossy re-saving.** Never re-encode images lossily after download. Store derived images as PNG or NPZ.

---

## 2. How this differs from LITHOS, and what to reuse or remove from the LITHOS code

### 2.1 Conceptual differences

| | LITHOS (Chacón et al., NeurIPS 2025) | RotaPetro (this project) |
|---|---|---|
| Input per grain | one 256×256 PPL patch + one XPL patch at 0° | 36 angles × (PPL + XPL) of the grain |
| Labels | expert point-counting, about one person-year | automatic, from rotation physics + OU mineral lists; zero manual labels |
| Rocks | Colombian sedimentary (hydrocarbon wells) | UK igneous, metamorphic and sedimentary (UKVM) |
| Output | a class for the patch | mineral + group + optical properties + a text explanation; field summary |
| Unit | patch centred on an annotated point | segmented grain (region) with its mask |
| Main result | 25-class macro-F1 ≈ 0.43 (static) | rotation vs 0° gain; accuracy vs number of angles |

### 2.2 The LITHOS repository (reviewed file by file)

`LITHOS-main/` contains these files:
- `main.py` — CLI, model zoo, training and validation loop, wandb logging
- `data_loader.py` — `LithosDataset`: reads LITHOS CSVs and 0° PPL/XPL patches
- `polar_vit_arch.py` — `PolarViT`: two ViT-L/16 encoders, cross-attention decoders, α-fusion
- `utils.py` — `AverageMeter`, `get_lr`, `save_epoch`
- `meters/` — logging helpers, parameter/FLOP/memory utilities (fvcore)
- `environment.yaml` — conda environment (CUDA 12.4, torch)

### 2.3 Remove (do not carry into RotaPetro)
- **`LithosDataset`.** It depends on LITHOS files: the `Sec_patch` column, `center_component.csv` and the `ppl-0` / `xpl-0` naming. It also uses a hard-coded 2/25-class remap (`remap_dict`). Replace it with our `RegionSequenceDataset` (§13.3).
- **The binary Quartz vs Non-Quartz task** and the fixed 25-class list.
- **`v2.RandomRotation(degrees=45)` augmentation.** Arbitrary rotation changes the angle between the crystal and the polariser without changing the colours, which is physically impossible in XPL/PPL. Use the physics-consistent augmentations in §15.4 instead.
- **2-fold CV with `assert args.fold in [1, 2]`.** We use 5-fold, grouped by specimen (§13.4).
- **ViT-L/16 as the default backbone.** It has about 300M parameters per encoder, ~600M in PolarViT, and is too heavy for our data size. Keep it only as an optional baseline (§14.7).
- **Hard dependency on wandb.** Make it optional; log to CSV and JSON by default.

### 2.4 Reuse (copy and adapt)
- **`utils.AverageMeter`, `utils.get_lr`, `utils.save_epoch`**: checkpointing pattern with best and last checkpoints.
- **`meters/logging.py`, `meters/misc.py`**: logging setup, parameter count, GPU memory.
- **The skeleton of `train()` / `val()` in `main.py`**: loop structure, softmax, macro precision/recall/F1, one-vs-rest ROC-AUC, per-class PR-AUC. Extend it for our extra heads and metrics.
- **`polar_vit_arch.Decoder` / `MultiHeadAttention`**: optional reuse as a PPL↔XPL cross-attention block (§14.6, variant V3).
- **`PolarViT` as a whole**: optional static baseline B3 on our 0° crops (§14.7).

**Bug to know about if you reuse `main.py`:** `--balance_loss` / `--balance_dataloader` call `train_ds._get_weights_class()`, which is not defined in `LithosDataset`. These flags crash as released. Implement class weights yourself (§15.2).

---

## 3. Repository layout and file locations

Create this structure. All paths in this document are relative to the repository root `rotapetro/`.

```
rotapetro/
├── README.md                     ← short version of this spec + quickstart
├── PROJECT_SPEC.md               ← this file (ROTAPETRO_PROJECT_SPEC.md)
├── configs/
│   ├── default.yaml              ← all paths and thresholds (§22.A)
│   └── experiments/*.yaml        ← one file per training experiment
├── data/
│   ├── raw/ukvm_dataset/         ← downloader output, never modified (§5)
│   ├── interim/
│   │   ├── sites.csv             ← cleaned list of unique rotation sites (§6)
│   │   ├── specimens_clean.csv   ← rebuilt specimen table (§6)
│   │   ├── aligned/<site_id>.npz ← aligned 36-angle stacks (§7)
│   │   ├── maps/<site_id>.npz    ← per-pixel property maps (§8)
│   │   ├── segments/<site_id>.png, <site_id>.json   ← region label image + list (§9)
│   │   └── qc/…                  ← QC images (PNG) for every phase
│   └── processed/
│       ├── regions.parquet       ← one row per region: geometry + property card (§10)
│       ├── labels.parquet        ← automatic labels and weights (§12)
│       ├── classes.yaml          ← final class list and hierarchy (§12.6)
│       ├── splits/folds.json     ← 5 specimen-grouped folds (§13.4)
│       └── embeddings/<encoder>/…← frozen-encoder features (§14.2)
├── knowledge/
│   ├── minerals.csv              ← the textbook table (§11)
│   ├── aliases.csv               ← OU mineral names → canonical names (§11.4)
│   ├── hierarchy.yaml            ← mineral groups (§11.5)
│   ├── likelihoods.yaml          ← P(observation | mineral) settings (§12.2)
│   ├── michel_levy_lut.npz       ← generated interference-colour table (§8.5)
│   ├── assemblage_hints.yaml     ← mineral assemblage → rock-family hints (§17.6)
│   └── SOURCES.md                ← which book/database each row came from
├── src/rotapetro/
│   ├── io.py                     ← readers/writers for all formats
│   ├── clean.py                  ← Phase 0
│   ├── align.py                  ← Phase 1
│   ├── maps.py                   ← Phase 2
│   ├── michel_levy.py            ← interference colour simulation (§8.5)
│   ├── segment.py                ← Phase 3
│   ├── features.py               ← Phase 4
│   ├── knowledge.py              ← Phase 5 loading/validation
│   ├── rules.py                  ← Phase 6 rule engine
│   ├── dataset.py                ← Phase 7 crops, splits, torch Dataset
│   ├── embed.py                  ← frozen encoder feature extraction
│   ├── model.py                  ← RotaPetro-Net and baselines
│   ├── train.py                  ← Phase 9 (reuses LITHOS loop pattern)
│   ├── evaluate.py               ← Phase 10
│   ├── infer_video.py            ← Phase 11
│   ├── explain.py                ← text explanations
│   └── utils/ (logging, meters, seed, plotting)
├── scripts/
│   ├── 00_clean.py … 12_infer.py ← thin CLI wrappers, one per phase (§20)
├── tools/
│   ├── ukvm_download.py          ← the downloader we already have
│   └── physics_maps.py           ← prototype of Phases 1–2 (reference only)
└── outputs/
    ├── models/<exp>/fold<k>/seed<s>/{best.pt, last.pt, config.yaml, metrics.json}
    ├── predictions/<exp>/…       ← per-region predictions for every test fold
    └── reports/                  ← tables and figures for the paper
```

**Identifiers (use them everywhere):**
- `code`: the OU asset code, e.g. `11agt25`.
- `site_id` = `<code>_r<k>`. `k` is renumbered 1..n after removing duplicates, e.g. `11agt25_r1`.
- `region_id` = `<site_id>_g<nnn>`, e.g. `11agt25_r1_g007`.
- **Angles:** `theta_deg` is the stage angle of a frame relative to frame 000, in degrees.

---

## 4. Environment and hardware

**Python 3.11.** `requirements.txt`:

```
numpy>=1.26
scipy>=1.11
pandas>=2.1
pyarrow>=14
opencv-python>=4.9
scikit-image>=0.22
scikit-learn>=1.4
lightgbm>=4.3
colour-science>=0.4.4
torch>=2.2
torchvision>=0.17
pyyaml
tqdm
matplotlib
requests
beautifulsoup4
pillow
# optional
segment-anything          # SAM, optional segmentation refinement (§9.6)
wandb                     # optional logging
imageio[ffmpeg]           # video decoding for inference
```

**DINOv2 encoder:** `torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')`. It needs internet the first time and is cached afterwards.

**Hardware:**

| Phase | Where it runs | Rough time for ~250 sites |
|---|---|---|
| 0–6 | laptop CPU (8 GB+ RAM) | Phase 1 ≈ 30–60 min; Phase 2 ≈ 5 min; Phases 3–4 ≈ 30–90 min; Phase 6 < 5 min |
| Embedding extraction (§14.2) | one GPU (Colab/Kaggle T4, or PARAM Sanganak) | ≈ 15–40 min |
| Training | GPU or CPU | minutes per fold |

**Disk:** raw download ≈ 30 GB with whole slides; aligned stacks ≈ 5 GB; embeddings ≈ 2–5 GB.

---

## 5. Raw data (what the downloader produced)

Location: `data/raw/ukvm_dataset/`. It is produced by `tools/ukvm_download.py --collections uk-virtual-microscope`. Never edit files in `raw/`.

```
data/raw/ukvm_dataset/
├── specimen_index.json      {code: {"node": "...", "collections": [...]}}
├── specimens.csv            one row per specimen (see known issues)
├── rotation_fields.csv      one row per downloaded rotation point (contains duplicates)
├── failures.csv             specimens where something was not found
└── specimens/<code>/
    ├── meta.json            all metadata (below)
    ├── rot/r<k>/ppl/000.jpg … 071.jpg   frame i = stage angle 5·i degrees (0…355)
    ├── rot/r<k>/xpl/000.jpg … 071.jpg   same angles, perfectly paired with PPL
    └── wsi/ppl.png, xpl.png (+ *_info.json, *_ImageProperties.xml)   whole slide at 0° (not used in core)
```

**Facts measured on the data** (granite and 11agt25):
- Frames are 490×490 px, RGB JPEG.
- The 72 frames cover 0–355° in exact 5° steps: measured 0, 5.2, 10.2, … 355.2°.
- Rotation is about the image centre, at about (244.2, 244.4) px, with scale 1.000.
- The PPL and XPL frames at the same index are co-registered to within 0.5 px.
- The field of view is a circle of radius ≈ 235 px; outside it is white.
- **Pixel size ≈ 4.0 µm/px.** This comes from the viewer's 1 mm scale bar (234 px on a 460 px display) and must be confirmed in Phase 0 (§6.5).
- **Whole slides were saved at zoom level 5 of 0–6**, i.e. half of full resolution (e.g. 7588×5598 of 15175×11195). They are not needed for the core pipeline.

**`meta.json` keys:**
- `code`, `node`, `title`, `collection`, `type`, `category` (sometimes missing)
- `rock_forming_mineral` (list), `accessory_minerals` (list)
- `description`, `locality`, `lat`, `lon`, `owner`, `licence`
- `rotation_fields` (list), `whole_slide` (dict), `problems`

**Known issues in the download (Phase 0 fixes them):**
1. **Duplicate rotation points.** The downloader read the viewer settings from two pages, so some specimens list every site twice. Example: `11agt25` has r3 ≡ r1 and r4 ≡ r2, byte-identical files.
2. **Duplicate whole-slide entries** (`PPL`, `XPL`, `ppl`, `xpl`). Only two real images exist on disk.
3. **Empty `accessory_minerals` column in `specimens.csv`.** The table writer read the key `accessory_mineral`, but the metadata key is `accessory_minerals`. Rebuild the table from `meta.json`.
4. **Missing `category` for some specimens.** Use `title` + `type` instead.

---

## 6. Phase 0 — Clean and audit the download

Script `scripts/00_clean.py` (module `clean.py`). Input: `data/raw/ukvm_dataset/`. Outputs: `data/interim/sites.csv`, `data/interim/specimens_clean.csv`, `data/interim/qc/phase0_report.md`.

### 6.1 Remove duplicate rotation sites
For every specimen, for every `rot/r*`:
1. Compute `h = md5(ppl/000.jpg) + md5(xpl/000.jpg) + md5(ppl/037.jpg)`.
2. Keep the first site with each hash; mark the others `dup_of = <kept site>`.
3. Renumber the kept sites r1..rn in their original order.
4. Do not delete raw folders. `sites.csv` records the mapping `raw_folder → site_id`.

### 6.2 Completeness checks (per site)
- 72 PPL and 72 XPL frames; all files decode; all frames 490×490. Otherwise `usable = False`.
- Record the mean brightness of every frame. Flag `exposure_jump` if the PPL mean inside the circle changes by more than 15% between consecutive frames. Pleochroic grains change it smoothly; jumps mean camera problems.

### 6.3 Rebuild the specimen table
Write `specimens_clean.csv` from all `meta.json` files, with these columns:
- `code`, `node`, `title`, `type`, `category`
- `rock_forming` (semicolon list), `accessory` (semicolon list)
- `n_sites_unique`, `locality`, `lat`, `lon`, `owner`
- `is_monomineralic` (§6.4)
- `rock_family` (manual mapping from title/category, e.g. granitoid, basalt, gabbro, schist, gneiss, slate, sandstone, limestone, marble, quartzite…)

### 6.4 Flag monomineralic specimens (free grain-level truth)
`is_monomineralic = True` if either condition holds:
- `rock_forming` contains exactly one mineral after alias mapping (§11.4); or
- the title or category is one of: quartzite, marble, dolostone, dunite, anorthosite, chert, pure limestone.

A physics sanity check comes later (§12.4). These specimens give grain-level ground truth without humans.

### 6.5 Rotation-site titles
The viewer shows a title above each rotation view; `microscope.js` sets it from the `title` field of each rotation entry.
1. Open `https://virtualmicroscope.org/sites/default/files/html5Assets/<code>/index.html` for 2–3 specimens (view-source).
2. Find the structure that contains `pplData` / `xplData` and the rotation `title`.
3. Write a small parser that extracts `(order, title, pplData, xplData)` for every rotation, deduplicating by `pplData`.
4. Map titles to sites through `pplData` (the frame URL prefix), not through order.
5. If a title names a mineral (e.g. "tourmaline crystal"), store it in `sites.csv` as `title_minerals`, mapped through `aliases.csv`.

These titles later give a small grain-level test set (T3, §16.3). If titles are generic (e.g. "Rotation 1"), leave the column empty.

### 6.6 Confirm the pixel size (one-time, 10 minutes)
Rotation frames and whole slides share the same physical scale, so one measurement calibrates everything:
1. Open the viewer for 3 specimens and read the physical width of the whole slide from the mm ruler.
2. Compute `um_per_px_full = width_mm*1000 / full_width_px`, with `full_width_px` from `wsi/*_info.json`.
3. Register rotation frame 000 onto the whole slide. `tools/physics_maps.py` style ORB/SIFT matching works; for 11agt25 the frame→level-5 scale was 0.5686.
4. Then `um_per_px_rot = um_per_px_level5 * scale_frame_to_level5`, where `um_per_px_level5 = 2 * um_per_px_full`.
5. Write the median value to `configs/default.yaml` as `um_per_px_rot`. Default if not measured: 4.0.

### 6.7 Phase 0 report (must be read before continuing)
Write `phase0_report.md` listing:
- number of specimens, unique sites, usable sites, duplicates removed;
- number of monomineralic specimens;
- titles found;
- frequency of every mineral name in the OU lists (input for `aliases.csv`);
- rock families and their counts.

---

## 7. Phase 1 — Align the rotation frames

Script `scripts/01_align.py` (module `align.py`). Input: `sites.csv` (usable sites) + raw frames. Output: `data/interim/aligned/<site_id>.npz`, `data/interim/qc/align/<site_id>.png`.

**Why:** the stage rotates, so a grain moves around the image from frame to frame. We rotate every frame back to the orientation of frame 000 so that each grain stays on the same pixels. After alignment, the only thing that changes between frames is the angle of the polarisers relative to the crystal, which is exactly the optical signal we want.

### 7.1 Algorithm (per site)
1. Read all 72 PPL and 72 XPL frames as uint8 RGB.
2. **Reference:** PPL frame 000 (grayscale). Detect ORB features (`nfeatures=3000`) inside the circle (mask radius 230 px).
3. **For each frame i = 1..71 (PPL, grayscale):**
   1. Detect ORB features.
   2. Match to the reference with a Hamming brute-force matcher and cross-check; keep the best 600 matches.
   3. Estimate a similarity transform with `cv2.estimateAffinePartial2D(frame_pts, ref_pts, RANSAC, ransacReprojThreshold=2)`.
   4. Read the measured rotation `rot_i = atan2(M[1,0], M[0,0])` (degrees) and the scale `s_i = hypot(M[0,0], M[1,0])`.
4. **Sanity check:** `|rot_i − (±5·i)| ≤ 1.5°` (mod 360) and `|s_i − 1| ≤ 0.01`. Determine the sign once from frame 1.
   - If it fails, retry by registering to frame i−1 and composing the transforms.
   - If it still fails, use the nominal rotation `5·i°` about the median rotation centre of the site. Set `align_fallback = True`.
5. **Warp** the PPL frame and the XPL frame i with the **same** matrix M: `cv2.warpAffine`, bilinear, border value 0.
6. **Valid mask:** pixels inside a circle of radius 232 px around the rotation centre, and non-zero in every warped frame.
7. **Fold 360° into 180°.** Optical behaviour repeats every 180° of stage rotation (PPL has a 180° period, XPL a 90° period). So average frame i with frame i+36 (both already aligned) to get **36 frames at 0, 5, …, 175°** with lower noise. Store the float mean rounded back to uint8.

### 7.2 Output format `aligned/<site_id>.npz`

| key | dtype / shape | meaning |
|---|---|---|
| `ppl` | uint8 [36, 490, 490, 3] | aligned, folded PPL frames, RGB |
| `xpl` | uint8 [36, 490, 490, 3] | aligned, folded XPL frames, RGB |
| `theta_deg` | float32 [36] | 0, 5, …, 175 |
| `valid` | bool [490, 490] | usable pixels |
| `M` | float32 [72, 2, 3] | transform of every raw frame to frame 000 |
| `rot_measured_deg` | float32 [72] | measured rotation per raw frame |
| `rot_sign` | int8 | +1 or −1 (rotation sense in image coordinates) |
| `center_xy` | float32 [2] | rotation centre |
| `align_fallback` | bool [72] | frames that used the nominal transform |

### 7.3 QC and definition of done
Save a PNG with PPL frames 0°, 45°, 90° and 135° side by side plus a flicker difference image.

Pass criteria:
- median angle error < 0.5°;
- fewer than 3 fallback frames;
- mean absolute difference between aligned PPL frames 000 and 036 (before folding) < 6 grey levels inside `valid`.

If there are more than 5% failures overall, inspect them before continuing.

---

## 8. Phase 2 — Optical property maps (per pixel)

Script `scripts/02_maps.py` (modules `maps.py`, `michel_levy.py`). Input: aligned stacks. Output: `data/interim/maps/<site_id>.npz` and `qc/maps/<site_id>.png` (6-panel figure, as in the prototype `tools/physics_maps.py`).

Notation: N = 36 angles, θ_k = 5k° (k = 0..35), in radians inside formulas. All arrays are computed only on `valid` pixels. The background is NaN.

### 8.1 Physics being used
- **XPL:** a birefringent grain between crossed polars transmits `I(θ) = I_bg + A · sin²(2(θ − θ₀))`. It goes dark (extinct) every 90°, when a vibration direction is parallel to the polariser. θ₀ is the extinction position; A depends on the retardation Γ = d·δ (thickness × birefringence).
  Identity: `sin²(2x) = (1 − cos 4x)/2`, so the signal lives in the **4th harmonic** of θ.
- **PPL:** a pleochroic grain's colour changes with a 180° period: `I_c(θ) = a_c + b_c · cos(2(θ − θ₁))` per colour channel c, which is the **2nd harmonic**.
- **Isotropic / opaque:** an isotropic grain (garnet, glass) is dark in XPL at every angle. An opaque grain is dark in both PPL and XPL at every angle.

### 8.2 Luminance and harmonics
For every pixel and angle:
- `L_xpl[k] = 0.299R + 0.587G + 0.114B` of the XPL frame (float, 0–255)
- `L_ppl[k]` likewise for the PPL frame

Discrete Fourier coefficients over the 36 angles:
- `m0 = mean_k L_xpl[k]`
- `c4 = (2/N) Σ L_xpl[k]·cos(4θ_k)`, `s4 = (2/N) Σ L_xpl[k]·sin(4θ_k)`
- `A4 = hypot(c4, s4)`, `φ4 = atan2(s4, c4)`

Derived maps:
- **Extinction position** `ext_deg = degrees(((φ4 + π)/4) mod (π/2))`, in [0, 90). This is the stage angle of the first extinction.
- **Anisotropy index** `aniso = clip(A4 / (m0 + 1), 0, 1)`. It is near 0 for isotropic, opaque and hole pixels, and high for birefringent grains.
- **Fit quality** `r2_4 = (A4²/2) / var_k(L_xpl[k])`: the fraction of the XPL variance explained by the 4th harmonic, in 0–1. It is low for noise, very small grains and aggregates.
- **XPL brightest frame** `k_max`: the angle index nearest to `ext_deg + 45°`. Define `xpl_max_rgb` as the mean RGB of frames at `ext + 45°` and `ext + 135°` (two frames).
- `xpl_maxproj_rgb` = per-pixel maximum over all 36 XPL frames (used for segmentation).

### 8.3 PPL colour, pleochroism, opacity, relief change
- `ppl_mean_rgb` = mean over k; `ppl_min_L` = minimum over k of `L_ppl`; `ppl_max_L` = maximum.
- **2nd harmonics:** for every colour channel c compute `a_c = mean_k I_c[k]`, `c2_c = (2/N) Σ I_c[k]·cos(2θ_k)` and `s2_c = (2/N) Σ I_c[k]·sin(2θ_k)`. Compute the same for `L_ppl`.
  - **Absorption angle:** `pleo_dark_deg` = the stage angle (0–180°) where the fitted `L_ppl` is lowest: `θ_dark = ((atan2(s2_L, c2_L) + π)/2) mod π`.
  - **Extreme colours:** evaluate each channel's fitted curve `a_c + c2_c·cos 2θ + s2_c·sin 2θ` at `θ_dark` and at `θ_dark + 90°`. These are the two pleochroic colours (`col_dark`, `col_light`). Convert both to CIELAB.
  - **Pleochroism strength:** `pleo_dE = ΔE76(col_dark, col_light)`.
- **PPL colour in Lab** (`ppl_Lab`) of `ppl_mean_rgb` after white balance (§8.4).
- **Opacity:** `opaque_px = ppl_max_L < 0.15·W` (W = white level, §8.4).
- **Relief change ("twinkling")** of calcite, dolomite and muscovite. Compute the gradient magnitude `G[k]` of `L_ppl[k]` (Sobel, 3×3, after Gaussian σ=1). Then `relief_mod = 2nd-harmonic amplitude of G[k] / (mean_k G[k] + 1)`, smoothed over a 7×7 window.
- **PPL texture:** `ppl_tex` = local standard deviation (7×7) of the high-pass `L_ppl` (mean over k). This is a relief/"shagreen" and alteration proxy.

### 8.4 White level and white balance (per site)
- **White level W:** the 99th percentile of `ppl_max_L` inside `valid`. It approximates the lamp through clear material or holes.
- **White balance:** take the pixels with `ppl_max_L > 0.9·W` and PPL chroma < 10. Compute per-channel means (R̄, Ḡ, B̄) and gains `g_c = mean(R̄, Ḡ, B̄)/c̄`. Apply the gains to all PPL and XPL colours before any Lab conversion.
- **Relative XPL colour:** `xpl_rel_rgb = xpl_max_rgb / max(ppl_mean_rgb, 1)` per channel, clipped to [0, 1.5]. This divides out the lamp colour and the absorption of coloured minerals, giving ≈ the fraction of light transmitted between crossed polars at 45°, which is what the Michel-Lévy chart predicts.

### 8.5 Birefringence from colour (Michel-Lévy lookup)
**Generate `knowledge/michel_levy_lut.npz` once**, with `colour-science`:
1. For retardation Γ = 0…3000 nm in 2 nm steps, compute the spectrum `S(λ) = D65(λ) · sin²(π Γ / λ)` for λ = 380–780 nm, 5 nm steps. This is transmission at 45° between crossed polars.
2. Convert to XYZ with the CIE 1931 2° colour-matching functions, normalising so that the XYZ of `D65(λ)` alone has Y = 1.
3. Convert to linear sRGB, then to CIELAB. Store `gamma_nm [1501]`, `rgb [1501,3]` and `lab [1501,3]`.
4. **Check:** Γ ≈ 550 nm must look first-order red/magenta, Γ ≈ 270 nm pale yellowish-white, and Γ > 2000 nm near-white pastel.

**Per pixel (anisotropic pixels only, `aniso ≥ 0.15`):**
- `gamma_est_nm` = argmin over Γ ∈ [0, 2000] of `ΔE76(Lab(xpl_rel_rgb), lab_LUT(Γ))`. Also store `gamma_dE`, the matching error.
- **High-order white rule:** if `gamma_dE > 25` AND the observed colour has L* > 75 and chroma < 15 AND `relief_mod` is high (> 0.3), set `gamma_est_nm = 2500` (class C4).

**Birefringence class `biref_cls`** from `gamma_est_nm`:

| class | Γ range (nm) | appearance at 30 µm | typical minerals |
|---|---|---|---|
| C0 | < 100 or not anisotropic | black / dark grey | isotropic, apatite, nepheline (very low) |
| C1 | 100–350 | grey, white, pale yellow (1st order) | quartz, feldspars, cordierite, chlorite |
| C2 | 350–1100 | yellow–orange–red–blue (1st–2nd order) | orthopyroxene, hornblende, augite, kyanite, epidote |
| C3 | 1100–2000 | bright 2nd–3rd order | olivine, muscovite, zircon |
| C4 | > 2000 | high-order pastel "white" | calcite, dolomite, titanite |

**Caveat:** JPEG colours, unknown section thickness and camera response make Γ approximate (±30%). Use classes, not exact nm, in the rules.

### 8.6 Output format `maps/<site_id>.npz`

| key | dtype | meaning |
|---|---|---|
| `ext_deg` | float16 [H,W] | extinction position 0–90° |
| `aniso` | float16 | anisotropy index 0–1 |
| `r2_4` | float16 | fit quality 0–1 |
| `xpl_max_rgb`, `xpl_rel_rgb`, `xpl_maxproj_rgb` | uint8 / float16 [H,W,3] | see §8.2, §8.4 |
| `gamma_est_nm`, `gamma_dE` | float16 | §8.5 |
| `biref_cls` | int8 | 0–4 |
| `ppl_mean_rgb` | uint8 [H,W,3] | white-balanced mean PPL |
| `ppl_Lab` | float16 [H,W,3] | Lab of mean PPL |
| `pleo_dE`, `pleo_dark_deg` | float16 | §8.3 |
| `ppl_min_L`, `ppl_max_L` | float16 | §8.3 |
| `relief_mod`, `ppl_tex` | float16 | §8.3 |
| `valid` | bool | copied from Phase 1 |
| `white_level`, `wb_gains` | float32 | §8.4 |

### 8.7 QC and definition of done
Render the 6-panel figure (PPL 0°, XPL 0°, XPL max projection, extinction colour map, anisotropy, pleochroism) for every site. Visually confirm on the granite and 11agt25:
- each grain shows one uniform extinction colour;
- microcline shows tartan stripes;
- biotite and tourmaline light up in the pleochroism map;
- quartz grains have `r2_4 > 0.6`.

---

## 9. Phase 3 — Automatic grain segmentation

Script `scripts/03_segment.py` (module `segment.py`). Input: aligned stacks + maps. Output:
- `segments/<site_id>.png`: uint16 label image; 0 = invalid/background, 1..n = regions;
- `segments/<site_id>.json`: region list with types;
- QC overlay PNG.

A **region** is one grain, or one fine-grained aggregate that cannot be split into grains.

### 9.1 Feature image for segmentation
Build an 8-channel image F (float32):
- channels 1–3: Lab of `xpl_maxproj_rgb`, each ×1.0
- channels 4–6: Lab of `ppl_mean_rgb`, ×0.7
- channel 7 and 8: `40 · aniso · cos(4·ext)` and `40 · aniso · sin(4·ext)`. This encodes extinction as a vector, so 0° and 89° are close (period 90°).

Smooth each channel with a 3×3 median filter.

### 9.2 Over-segmentation (superpixels)
Run `skimage.segmentation.slic(F, n_segments = valid_area/150, compactness=10, mask=valid, channel_axis=-1, start_label=1)`. This typically gives 1,000–1,500 superpixels per site.

### 9.3 Region merging (grain boundaries)
1. Build a region adjacency graph on F (`skimage.graph.rag_mean_color`, mode `distance`).
2. Merge hierarchically with threshold `T_merge = 14` (feature distance). Tune on 5 sites by eye so that one grain ≈ one region.
3. **Twin and sector merge (second pass).** Physics cue: twin and sector boundaries are nearly invisible in PPL, while true grain boundaries show as dark lines or relief steps in PPL. Merge adjacent regions A and B if all of these hold:
   - `ΔE(PPL Lab mean) < 6`, and
   - `|biref_cls_A − biref_cls_B| ≤ 1`, and
   - the mean PPL gradient along the shared boundary < 0.5 × the median PPL gradient along all region boundaries in the site, and
   - the shared boundary is ≥ 15 px long.

   Record `n_domains` = number of merged pieces, used later for twinning.
4. Remove regions < 150 px by merging them into the most similar neighbour.

### 9.4 Region types
- **`hole`**: `ppl_max_L > 0.92·W`, `aniso < 0.1`, `ppl_tex` in the lowest 10% of the site → epoxy, holes, slide background. **Excluded** from the dataset.
- **`opaque`**: > 80% of pixels are `opaque_px`.
- **`aggregate`**: area ≥ 400 px AND the median of the local (5×5) circular variance of `4·ext` > 0.6 AND `r2_4` median < 0.4. This means many tiny crystals with random orientation: sericite, chert, groundmass, clay.
- **`grain`**: everything else.
- **Flags:**
  - `partial = True` if the region touches the edge of `valid`. Shape features are then unreliable.
  - `small = True` if the area is < 400 px (≈ 80×80 µm at 4 µm/px). Small regions are kept for statistics but not used as training samples.

### 9.5 Output `segments/<site_id>.json`
```json
{"site_id": "11agt25_r1", "um_per_px": 4.0,
 "regions": [{"region_id": "11agt25_r1_g001", "label_value": 1, "type": "grain",
              "area_px": 5321, "bbox_xywh": [102, 88, 97, 81], "centroid_xy": [150.2, 128.9],
              "partial": false, "small": false, "n_domains": 1}]}
```

### 9.6 Optional refinement with SAM (v2)
Run SAM's automatic mask generator on `xpl_maxproj_rgb` and on `ppl_mean_rgb`. If a merged region crosses a strong SAM boundary that is also a strong PPL edge, split it. Only do this if the QC in §9.7 shows frequent under-segmentation.

### 9.7 QC and definition of done
Draw region boundaries on PPL 0° and XPL max for 20 random sites. Tune `T_merge` and the twin rule until obvious grains are single regions, while twinned plagioclase and microcline and the sector-zoned grain in 11agt25 r1 stay one region. No human labels are created; this is only parameter tuning.

---

## 10. Phase 4 — Grain property cards (features)

Script `scripts/04_features.py` (module `features.py`). Input: aligned stacks, maps, segments. Output: `data/processed/regions.parquet`, one row per region (all types except `hole`).

Robust statistics are computed over the region's pixels, excluding a 2-px band along the region border to avoid boundary effects. Weighted statistics use `aniso` as the weight where noted.

### 10.1 Identity and geometry

| feature | definition |
|---|---|
| `region_id, site_id, code, type, partial, small` | from Phase 3 |
| `area_px`, `area_um2` | pixel count; × (um_per_px)² |
| `eq_diam_um` | 2·sqrt(area/π) in µm |
| `major_px`, `minor_px`, `aspect` | from image moments (`skimage.measure.regionprops`); aspect = major/minor |
| `long_axis_deg` | orientation of the major axis, 0–180°, in aligned image coordinates. NaN if aspect < 1.3 or partial |
| `solidity`, `eccentricity` | regionprops |
| `habit` | equant (aspect < 1.5), prismatic (1.5–4), acicular (> 4); NaN if partial |

### 10.2 Optical class

| feature | definition |
|---|---|
| `opaque_frac` | fraction of pixels with `opaque_px` |
| `aniso_med`, `aniso_p90` | median / 90th percentile of `aniso` |
| `r2_4_med` | median fit quality |
| `optic_class` | `opaque` if opaque_frac > 0.8; `isotropic` if aniso_p90 < 0.15 and not opaque; else `anisotropic` |

### 10.3 Colour and pleochroism

| feature | definition |
|---|---|
| `ppl_L, ppl_a, ppl_b` | median of `ppl_Lab` |
| `ppl_chroma`, `ppl_hue_deg` | from a, b |
| `ppl_colour` | colourless if chroma < 6; else by hue: 20–75° brown, 75–110° yellow, 110–200° green, 200–290° blue, 290–20° pink/red. `black` if ppl_L < 20 |
| `pleo_dE_med` | median pleochroism ΔE |
| `pleo_cls` | none < 4, weak 4–10, moderate 10–20, strong > 20 (tune on biotite-bearing specimens) |
| `pleo_colours` | Lab of the two extreme colours (median over pixels) |
| `absorption_vs_long_deg` | smallest angle between the axis of maximum absorption and `long_axis_deg`, 0–90° (§10.6). Biotite ≈ 0°, tourmaline ≈ 90°. NaN if long axis undefined or pleo_cls = none |

### 10.4 Interference colour / birefringence

| feature | definition |
|---|---|
| `xpl_rel_L, a, b` | median Lab of `xpl_rel_rgb` |
| `gamma_med_nm`, `gamma_p90_nm` | median / 90th percentile of `gamma_est_nm` (anisotropic pixels) |
| `biref_cls` | mode of pixel classes weighted by aniso; C4 if > 30% of pixels are C4 |
| `relief_mod_med` | median twinkling index |
| `colour_masking` | True if ppl_chroma > 20 (strongly coloured minerals mask interference colours) |

### 10.5 Extinction, twinning, undulose extinction
Circular statistics use period 90°: `z = aniso · exp(i·4·ext)` for each pixel.

| feature | definition |
|---|---|
| `ext_mean_deg` | angle(mean z)/4, in 0–90 |
| `ext_R` | abs(mean z)/mean(aniso), 0–1 concentration |
| `ext_std_deg` | sqrt(−2 ln ext_R)/4, in degrees |
| `n_ext_domains` | number of clusters of `ext` inside the region: circular k-means (k = 1..3) on (cos 4ext, sin 4ext). A cluster counts only if it holds ≥ 10% of pixels and cluster centres differ by ≥ 8° |
| `twin_type` | none (1 domain); `simple` (2 domains, 1 boundary line); `lamellar` (≥ 3 alternating stripes with one dominant boundary orientation: the Hough transform of domain boundaries has one peak and the profile perpendicular to it alternates ≥ 3 times); `crosshatched` (two boundary orientations 70–110° apart, many small spindle domains); `sector` (≥ 3 domains meeting near one point) |
| `undulose_score` | median spatial gradient magnitude of the extinction angle (degrees per 10 px) inside a single domain. High with no discrete clusters = undulose |
| `ext_vs_long_deg` | smallest angle between an extinction direction and `long_axis_deg` (§10.6), 0–45°. NaN if long axis undefined |
| `ext_type` | parallel if ext_vs_long < 5°; inclined if 5–45°; undefined if NaN |

### 10.6 Converting stage angles to directions in the image
At stage angle θ (aligned coordinates), the polariser direction relative to the fixed sample is `p(θ) = p₀ − rot_sign·θ`, with p₀ = 0° (E–W polariser, image x-axis). A vibration direction of the crystal is where extinction occurs: `v = p₀ − rot_sign·ext_mean` (mod 90).

The extinction angle relative to the long axis is `min over n of |(v + n·90) − long_axis| (mod 180)`, folded into [0, 45]. The absorption axis is `p₀ − rot_sign·pleo_dark_deg` (mod 180).

**Calibrate sign and p₀ once.** Micas and sillimanite must give `ext_vs_long ≈ 0°`. Take all regions in mica-rich specimens with aspect > 2 and check the median. If it is ≈ 45°, the polariser is diagonal; if the median is wrong, flip `rot_sign`. Record the calibration in `configs/default.yaml`.

### 10.7 Cleavage
Inside the eroded region mask, on PPL mean luminance:
1. Apply a ridge filter for dark thin lines (`skimage.filters.sato`, black_ridges=True, sigmas 1–2).
2. Threshold at the 95th percentile of the site.
3. Skeletonise and run a probabilistic Hough transform (min line length 12 px).
4. Build an orientation histogram weighted by line length.

| feature | definition |
|---|---|
| `cleavage_strength` | ridge pixels on lines / region area |
| `n_cleavage_dirs` | number of histogram peaks holding ≥ 20% of line length (0, 1 or 2) |
| `cleavage_angle_deg` | angle between the two main directions (if 2), 0–90 |
| `ext_vs_cleavage_deg` | like ext_vs_long but relative to the main cleavage direction |
| `ext_bisects_cleavage` | True if the extinction direction bisects two cleavages within 5° (carbonates, amphibole basal sections) |

### 10.8 Relief and texture proxies
- `boundary_contrast`: mean PPL gradient on a 3-px band just inside the region border ÷ site median. High for high-relief minerals.
- `ppl_tex_med`: median `ppl_tex` (shagreen, cloudiness, alteration).
- `relief_cls`: low / moderate / high / very high by site-relative quartiles of `boundary_contrast`. **Weak evidence only.**

### 10.9 Output
`regions.parquet` holds all columns above, plus `feature_version` and `um_per_px`. **Checks:**
- no NaN in required columns for grain regions;
- the distribution plots of every feature (`qc/features/*.png`) look sensible: quartz-rich specimens peak at biref_cls C1, marbles at C4.

---

## 11. Phase 5 — Knowledge base (the "textbook" table)

Module `knowledge.py`. Files in `knowledge/`. This is the petrologist's knowledge in machine-readable form. It is used by the rule engine (labels and baseline), for explanations and at inference.

### 11.1 Where the knowledge comes from
Fill and verify every row from at least **two** of these sources, and record them in the `source` column and in `knowledge/SOURCES.md`:
- **Nesse, W.D.** — *Introduction to Optical Mineralogy* (Oxford University Press; any recent edition). Mineral-by-mineral optical data: colour, pleochroism, relief, birefringence, extinction, twinning, cleavage, distinguishing features.
- **Raith, M.M., Raase, P., Reinhardt, J.** — *Guide to Thin Section Microscopy* (2nd ed., 2012; free PDF distributed by the authors). Clear tables of optical properties and the Michel-Lévy chart.
- **Deer, W.A., Howie, R.A., Zussman, J.** — *An Introduction to the Rock-Forming Minerals* (Mineralogical Society). Authoritative numeric ranges.
- **MacKenzie, W.S., Adams, A.E.** — *A Colour Atlas of Rocks and Minerals in Thin Section*. Visual appearance and habits.
- **Online databases:** mindat.org ("Optical Data" section: refractive indices, birefringence δ, pleochroism, 2V) and webmineral.com (optical data). Use them for numeric δ ranges.

**Licensing rule:** store facts (numbers and categories) with citations, never copied paragraphs or figures.

### 11.2 Schema of `knowledge/minerals.csv`

| column | type | allowed values / meaning |
|---|---|---|
| `mineral` | str | canonical name, lowercase, snake_case (e.g. `k_feldspar` is a group, `orthoclase` a mineral) |
| `group` | str | parent group (§11.5) |
| `optic_class` | enum | `isotropic`, `uniaxial`, `biaxial`, `opaque` |
| `ppl_colour` | list | colourless, brown, red-brown, green, blue-green, blue, yellow, pink, purple, black, cloudy |
| `pleochroism` | list | none, weak, moderate, strong |
| `relief` | list | low, moderate, high, very_high, variable |
| `biref_min`, `biref_max` | float | δ range (dimensionless) |
| `colour_masks_biref` | enum | no, partial, yes (strong body colour hides interference colours) |
| `extinction` | list | parallel, inclined, symmetric, undulose, none |
| `ext_angle_max_deg` | float | maximum extinction angle to elongation/cleavage (blank if n/a) |
| `twinning` | list | none, simple, lamellar, crosshatched, sector |
| `cleavage_dirs` | int | 0, 1, 2 (4 for fluorite) |
| `cleavage_angle_deg` | str | e.g. `56-124`, `87-93`, `75-105` (blank if < 2) |
| `habit` | list | equant, prismatic, acicular, platy, fibrous, rhombic, tabular, anhedral, euhedral |
| `absorption_vs_long` | enum | `parallel` (darkest when elongation ∥ polariser, e.g. biotite), `perpendicular` (e.g. tourmaline), `n/a` |
| `twinkling` | enum | yes/no (strong relief change on rotation) |
| `special` | str | free-text diagnostic notes (used in explanations) |
| `look_alikes` | list | minerals commonly confused with it |
| `source` | str | citations |

### 11.3 Seed table (starting values — verify every row, then extend)
Values are typical ranges from standard optical mineralogy references. **Retardation at 30 µm = δ × 30 000 nm.**

```csv
mineral,group,optic_class,ppl_colour,pleochroism,relief,biref_min,biref_max,colour_masks_biref,extinction,ext_angle_max_deg,twinning,cleavage_dirs,cleavage_angle_deg,habit,absorption_vs_long,twinkling,special,look_alikes,source
quartz,silica,uniaxial,colourless,none,low,0.009,0.009,no,parallel;undulose,0,none,0,,anhedral;equant,n/a,no,"clear and unaltered; undulose extinction common; no cleavage",orthoclase;plagioclase;cordierite,TO_VERIFY
plagioclase,feldspar,biaxial,colourless;cloudy,none,low,0.007,0.013,no,inclined,40,lamellar;simple,2,86-94,tabular;anhedral,n/a,no,"albite (lamellar) twins; zoning; sericite alteration",orthoclase;quartz;cordierite,TO_VERIFY
orthoclase,k_feldspar,biaxial,colourless;cloudy,none,low,0.005,0.007,no,inclined;parallel,12,simple;none,2,88-92,anhedral;tabular,n/a,no,"cloudy/turbid alteration; perthite; Carlsbad twins",quartz;plagioclase;sanidine,TO_VERIFY
microcline,k_feldspar,biaxial,colourless;cloudy,none,low,0.006,0.008,no,inclined,20,crosshatched,2,88-92,anhedral,n/a,no,"tartan (cross-hatched) twinning; perthite",orthoclase;plagioclase,TO_VERIFY
sanidine,k_feldspar,biaxial,colourless,none,low,0.005,0.007,no,parallel;inclined,10,simple;none,2,88-92,tabular;euhedral,n/a,no,"clear phenocrysts in volcanic rocks",quartz;orthoclase,TO_VERIFY
muscovite,mica,biaxial,colourless,none,moderate,0.036,0.049,no,parallel,3,none,1,,platy,n/a,yes,"bright 2nd-3rd order colours; birds-eye extinction; twinkling",talc;phlogopite,TO_VERIFY
biotite,mica,biaxial,brown;red-brown;green,strong,moderate,0.030,0.070,yes,parallel,5,none,1,,platy,parallel,no,"darkest when cleavage parallel to polariser; pleochroic haloes around zircon; birds-eye extinction",hornblende;tourmaline,TO_VERIFY
chlorite,chlorite_serpentine,biaxial,green,weak;moderate,low;moderate,0.000,0.015,no,parallel,5,none;lamellar,1,,platy;fibrous,parallel,no,"anomalous berlin-blue or brown interference colours",biotite;serpentine,TO_VERIFY
serpentine,chlorite_serpentine,biaxial,colourless;green,none;weak,low,0.004,0.017,no,parallel,5,none,0,,fibrous,n/a,no,"mesh texture replacing olivine",chlorite,TO_VERIFY
hornblende,amphibole,biaxial,green;brown;blue-green,strong,moderate;high,0.014,0.034,partial,inclined,30,simple;none,2,56-124,prismatic,n/a,no,"two cleavages at 56/124 degrees in basal sections; diamond-shaped sections",augite;biotite;tourmaline,TO_VERIFY
actinolite,amphibole,biaxial,colourless;green,weak;moderate,moderate,0.017,0.027,no,inclined,20,simple;none,2,56-124,acicular;fibrous,n/a,no,"fibrous/acicular; tremolite is colourless",sillimanite;augite,TO_VERIFY
augite,pyroxene,biaxial,colourless;green;brown,none;weak,high,0.018,0.033,no,inclined,48,simple;lamellar;none,2,87-93,prismatic;equant,n/a,no,"cleavages near 90 degrees in basal sections; extinction 35-48 degrees",hornblende;olivine;orthopyroxene,TO_VERIFY
orthopyroxene,pyroxene,biaxial,colourless;pink;green,none;weak;moderate,high,0.007,0.020,no,parallel,3,none,2,88-92,prismatic,n/a,no,"parallel extinction; low 1st-order colours; pink-green pleochroism in hypersthene",augite;andalusite,TO_VERIFY
olivine,olivine,biaxial,colourless;green,none,high,0.035,0.052,no,parallel,3,none,0,,equant;euhedral,n/a,no,"irregular fractures; serpentine or iddingsite along cracks; bright 2nd-3rd order colours",augite;epidote,TO_VERIFY
garnet,garnet,isotropic,colourless;pink;brown,none,very_high,0.000,0.000,no,none,,none,0,,equant;euhedral,n/a,no,"black in XPL at every angle; high relief; fractured; inclusion-rich",spinel;fluorite;glass,TO_VERIFY
calcite,carbonate,uniaxial,colourless,none,variable,0.172,0.172,no,symmetric,,lamellar;none,2,75-105,anhedral;equant,n/a,yes,"extreme high-order creamy colours; strong twinkling; rhombohedral cleavage; e-twin lamellae",dolomite,TO_VERIFY
dolomite,carbonate,uniaxial,colourless;cloudy,none,variable,0.179,0.185,no,symmetric,,lamellar;none,2,75-105,rhombic;euhedral,n/a,yes,"rhombic crystals; cloudy cores; twin lamellae less common than calcite",calcite,TO_VERIFY
epidote,epidote,biaxial,yellow;green,weak;moderate,high,0.015,0.049,no,parallel,5,none;simple,1,,prismatic;equant,n/a,no,"pistachio colour; patchy anomalous bright colours",olivine;augite,TO_VERIFY
tourmaline,accessory,uniaxial,brown;green;blue,strong,moderate;high,0.016,0.035,partial,parallel,3,none,0,,prismatic;acicular,perpendicular,no,"darkest when elongation perpendicular to polariser; triangular basal sections; colour zoning",biotite;hornblende,TO_VERIFY
cordierite,al_silicate,biaxial,colourless,none,low,0.005,0.018,no,parallel;inclined,,sector;lamellar;none,0,,anhedral;equant,n/a,no,"sector/cyclic twins; yellow pleochroic haloes; pinite alteration",quartz;plagioclase,TO_VERIFY
staurolite,al_silicate,biaxial,yellow,moderate,high,0.010,0.015,partial,parallel,3,simple;none,0,,prismatic,n/a,no,"golden pleochroism; sieve texture with quartz inclusions; cross twins",andalusite;titanite,TO_VERIFY
kyanite,al_silicate,biaxial,colourless;blue,none;weak,high,0.012,0.016,no,inclined,30,lamellar;none,2,,prismatic;tabular,n/a,no,"bladed crystals; two cleavages; extinction up to about 30 degrees",sillimanite;andalusite,TO_VERIFY
sillimanite,al_silicate,biaxial,colourless,none,high,0.018,0.022,no,parallel,3,none,1,,fibrous;prismatic,n/a,no,"fibrolite mats; square basal sections with one diagonal cleavage",kyanite;actinolite,TO_VERIFY
andalusite,al_silicate,biaxial,colourless;pink,none;weak,high,0.009,0.012,no,parallel,3,none,2,88-92,prismatic,n/a,no,"chiastolite cross of inclusions; square sections",sillimanite;orthopyroxene,TO_VERIFY
titanite,accessory,biaxial,brown;yellow,weak;moderate,very_high,0.100,0.190,partial,inclined,,simple;none,1,,euhedral,n/a,no,"wedge (lozenge) shaped crystals; extreme birefringence partly masked",cassiterite;rutile,TO_VERIFY
zircon,accessory,uniaxial,colourless,none,very_high,0.042,0.065,no,parallel,3,none,0,,prismatic,n/a,no,"tiny prisms; causes pleochroic haloes in biotite",apatite;monazite,TO_VERIFY
apatite,accessory,uniaxial,colourless,none,moderate,0.001,0.007,no,parallel,3,none,0,,prismatic;acicular,n/a,no,"hexagonal basal sections; needles; dark-grey 1st-order colours",quartz,TO_VERIFY
cassiterite,accessory,uniaxial,brown;red-brown;yellow,weak;moderate,very_high,0.096,0.098,partial,parallel,3,simple;sector;none,0,,prismatic;equant,n/a,no,"colour zoning; twinning; very high relief",titanite;rutile,TO_VERIFY
nepheline,feldspathoid,uniaxial,colourless,none,low,0.003,0.005,no,parallel,3,none,0,,equant,n/a,no,"very low (dark grey) interference colours",quartz;sanidine,TO_VERIFY
glauconite,sheet_other,biaxial,green,moderate,moderate,0.014,0.032,yes,parallel,,none,0,,equant,n/a,no,"green pellets in sandstones",chlorite,TO_VERIFY
fluorite,isotropic_other,isotropic,colourless;purple,none,moderate,0.000,0.000,no,none,,none,4,70-110,equant,n/a,no,"isotropic; octahedral cleavage; negative relief",garnet;glass,TO_VERIFY
glass,isotropic_other,isotropic,colourless;brown,none,low;moderate,0.000,0.000,no,none,,none,0,,anhedral,n/a,no,"volcanic glass; no crystal shape; may contain microlites",garnet;fluorite,TO_VERIFY
opaque,opaque,opaque,black,none,n/a,,,n/a,none,,none,0,,equant;anhedral,n/a,no,"black in PPL and XPL at every angle (magnetite, ilmenite, pyrite, hematite...); needs reflected light",,TO_VERIFY
fine_aggregate,aggregate,biaxial,colourless;brown;cloudy,none;weak,low,0.000,0.050,partial,undulose,,none,0,,anhedral,n/a,no,"fine-grained mass of random crystals: sericite, clay, chert, groundmass",,DERIVED
```

**Rules for extending the table:**
- One row per mineral species you expect. Check the Phase 0 mineral-frequency list.
- Plagioclase extinction depends on composition (An%). Keep one plagioclase row for v1.
- Leave `ext_angle_max_deg` blank when no extinction angle is meaningful.

### 11.4 `knowledge/aliases.csv` (OU names → canonical)
Columns: `ou_name, canonical, level` (`mineral` or `group`). Build it from the Phase 0 frequency list. Examples:

```csv
ou_name,canonical,level
quartz,quartz,mineral
plagioclase,plagioclase,mineral
feldspar,feldspar,group
alkali feldspar,k_feldspar,group
k-feldspar,k_feldspar,group
orthoclase,orthoclase,mineral
microcline,microcline,mineral
mica,mica,group
biotite,biotite,mineral
muscovite,muscovite,mineral
sericite,fine_aggregate,mineral
pyroxene,pyroxene,group
clinopyroxene,augite,mineral
augite,augite,mineral
amphibole,amphibole,group
hornblende,hornblende,mineral
iron oxide,opaque,mineral
magnetite,opaque,mineral
ilmenite,opaque,mineral
pyrite,opaque,mineral
carbonate,carbonate,group
calcite,calcite,mineral
```

**Rule:** a group name in an OU list expands to all its member minerals when building candidate sets (§12.1).

### 11.5 `knowledge/hierarchy.yaml`
```yaml
felsic:
  silica: [quartz]
  feldspar:
    plagioclase: [plagioclase]
    k_feldspar: [orthoclase, microcline, sanidine]
  feldspathoid: [nepheline]
mica: [biotite, muscovite]
chlorite_serpentine: [chlorite, serpentine]
amphibole: [hornblende, actinolite]
pyroxene: [augite, orthopyroxene]
olivine: [olivine]
garnet: [garnet]
carbonate: [calcite, dolomite]
epidote: [epidote]
al_silicate: [kyanite, sillimanite, andalusite, staurolite, cordierite]
accessory: [tourmaline, titanite, zircon, apatite, cassiterite]
sheet_other: [glauconite]
isotropic_other: [glass, fluorite]
opaque: [opaque]
aggregate: [fine_aggregate]
```

The loader must validate that every mineral in `minerals.csv` appears exactly once in the hierarchy, and that every alias target exists.

---

## 12. Phase 6 — Automatic labels (no human labelling)

Script `scripts/06_autolabel.py` (module `rules.py`). Inputs: `regions.parquet`, `specimens_clean.csv`, `sites.csv`, `knowledge/*`. Output: `data/processed/labels.parquet`, `data/processed/classes.yaml`, `qc/labels/*.png` (contact sheets per label for visual sanity).

There are four independent label sources. Only the first is used for training labels by default; sources 2–3 are mainly for evaluation.

| source | what | used for |
|---|---|---|
| **S1. Rule engine** | property card + knowledge base, restricted to the specimen's OU minerals | training labels (confident only), baseline B0 |
| **S2. Monomineralic specimens** | every grain of a one-mineral rock gets that mineral | evaluation set T2 (+ training in other folds) |
| **S3. Rotation titles** | the region at the centre of a site whose title names a mineral | evaluation set T3 |
| **S4. OU mineral list** | set of minerals present in the specimen | weak (bag-level) training signal L_MIL + evaluation T1 |

### 12.1 Candidate set per specimen
- `C_s` = the union of the canonical minerals of the specimen's `rock_forming` + `accessory` lists, with group names expanded to their members.
- **Always add** `opaque` and `fine_aggregate`: they are often not listed but nearly always present.
- **Unconstrained mode** (for baselines and inference without hints) uses all minerals in the table.

### 12.2 Rule engine: scoring a region against a mineral
For region r and mineral m ∈ C_s, compute

`score(m | r) = Σ_f w_f · log P_f(obs_f(r) | m) + log prior(m)`

using only features that are defined (not NaN) for r. The prior is uniform over C_s. The terms `P_f`, with weights w_f and default likelihoods (all configurable in `knowledge/likelihoods.yaml`):

| f | observation | P(obs \| m) default | w_f |
|---|---|---|---|
| optic class | `optic_class` | 0.90 if equal to m's class (uniaxial/biaxial both count as anisotropic), else 0.03 | 2.0 |
| birefringence | `biref_cls` | expected classes of m = classes intersecting [δ_min·25 000, δ_max·35 000] nm. 0.70 if observed ∈ expected; 0.20 if adjacent; 0.03 otherwise | 1.5 (×0.3 if `colour_masks_biref = yes`, ×0.6 if `partial`) |
| pleochroism | `pleo_cls` | 0.75 if listed for m; 0.20 if adjacent level; 0.05 otherwise | 1.2 |
| PPL colour | `ppl_colour` | 0.70 if listed; 0.15 if m is "colourless" and observed chroma < 10; 0.05 otherwise | 1.0 |
| twinning | `twin_type` | 0.70 if listed; `none` observed for a mineral with lamellar/crosshatched twins → 0.35 (twins are not always visible); otherwise 0.05 | 1.2 |
| extinction type | `ext_type` | 0.75 if listed (for inclined, also require ext_vs_long ≤ ext_angle_max + 5°); 0.15 otherwise | 1.0 (only if aspect ≥ 1.5 or n_cleavage_dirs ≥ 1) |
| undulose | `undulose_score` high | quartz 0.6; others 0.15 | 0.5 |
| cleavage | `n_cleavage_dirs`, `cleavage_angle_deg` | 0.6 if dirs match; +0.2 if the angle falls in the listed range; 0.2 otherwise | 0.6 |
| absorption axis | `absorption_vs_long_deg` | biotite-type: 0.8 if < 20°; tourmaline-type: 0.8 if > 70°; else 0.2 | 1.0 (only if pleo ≥ moderate and aspect ≥ 1.5) |
| twinkling | `relief_mod_med` high | 0.8 if m twinkles; 0.2 otherwise | 0.8 |
| habit | `habit` | 0.6 if listed; 0.3 otherwise | 0.4 |
| relief | `relief_cls` | 0.5 if listed; 0.3 otherwise | 0.3 |
| region type | `aggregate` | fine_aggregate 0.9; others 0.05 | 2.0 |

Convert to a posterior with a softmax over `score(m|r)` for m ∈ C_s. Also store the **evidence**: the per-feature contributions `w_f·log P_f` for the top 2 minerals, used for explanations.

### 12.3 Acceptance rules (confidence)
A region gets a **fine label** if all of these hold:
- p_top ≥ 0.80 and (p_top − p_second) ≥ 0.40;
- type = grain or opaque, and not small;
- r2_4_med ≥ 0.4 (except isotropic or opaque);
- at least 4 non-NaN features.

Otherwise, a **group label** if the top candidates share a group whose summed posterior is ≥ 0.85 (e.g. "k_feldspar", "feldspar", "mica").

Otherwise the region is **unlabelled** (still used for MIL and self-training).

`fine_aggregate` regions are labelled `fine_aggregate` if their type is aggregate and `fine_aggregate` ∈ C_s (it always is).

### 12.4 Monomineralic specimens (S2)
- For specimens flagged in §6.4, every grain region gets the specimen's mineral, **unless** the physics clearly contradicts it (e.g. optic_class opaque in a quartzite). In that case label it `opaque` or leave it unlabelled.
- **Sanity check:** at least 70% of the grain area must be consistent with the mineral's optic class and birefringence class. Otherwise remove the monomineralic flag and log it.
- `label_source = mono`.

### 12.5 Rotation titles (S3)
If `title_minerals` has exactly one mineral:
1. Find the region containing the image centre (245, 245), or the largest region within 40 px of it.
2. Label that region with the mineral, `label_source = title`.

Use S3 **only for evaluation** (T3).

### 12.6 Final class list (`classes.yaml`)
After labelling, count accepted fine labels per mineral and the number of distinct specimens they come from.
- A mineral becomes its own class if it has ≥ 30 labelled regions from ≥ 3 specimens.
- Otherwise its labels are moved to the parent group, and the group becomes a class if it passes the same test.
- Anything else becomes `other`.
- Keep `opaque` and `fine_aggregate` as classes.

Write both levels to `classes.yaml`:
```yaml
fine_classes: [quartz, plagioclase, k_feldspar, biotite, muscovite, chlorite, amphibole, augite, olivine, garnet, calcite, opaque, fine_aggregate, other]   # example
groups: {quartz: silica, plagioclase: feldspar, k_feldspar: feldspar, biotite: mica, ...}
```

### 12.7 Output `labels.parquet`

| column | meaning |
|---|---|
| `region_id` | key |
| `rule_top1`, `rule_p1`, `rule_top2`, `rule_p2` | rule-engine result (constrained by C_s) |
| `rule_top1_unconstrained`, `rule_p1_unconstrained` | same without C_s (for baselines) |
| `rule_evidence_json` | per-feature contributions for the top 2 |
| `label_fine` | final fine class or null |
| `label_group` | final group or null |
| `label_source` | rule, mono, title, none |
| `label_conf` | posterior used as training weight (mono = 1.0) |
| `in_candidate_set` | list C_s of the specimen (for MIL) |

**Expected coverage:** 40–70% of grain regions get a fine or group label. Contact sheets per class, 30 random regions each, must look plausible. Fix rules or likelihoods if a class sheet looks obviously wrong. That counts as tuning, not labelling.

---

## 13. Phase 7 — Training dataset assembly and splits

Module `dataset.py`. Script `scripts/07_build_dataset.py`.

### 13.1 Sample definition
One sample = one region with `type ∈ {grain, opaque, aggregate}` and `small = False`.

### 13.2 Crop extraction (done on the fly from the aligned stacks; crops are never stored)
For region r with bounding box (x, y, w, h) in site s:
1. Make the box square with side `S = max(w, h) × 1.3`, minimum 48 px, centred on the bbox centre. Clamp it to the image and pad with zeros if needed.
2. For every angle k = 0..35, crop the PPL and XPL frames and resize to **224×224** (bilinear). Also crop the region mask and resize it (nearest).
3. **Context dimming:** multiply pixels outside the region mask by 0.35, so the target grain is clear but its neighbours are still visible. Store the mask for pooling.

### 13.3 What the dataset returns (`RegionSequenceDataset.__getitem__`)
```python
{
  "region_id": str,
  "ppl": uint8 tensor [36, 3, 224, 224],   # only when images are needed (fine-tuning)
  "xpl": uint8 tensor [36, 3, 224, 224],
  "mask": bool tensor [224, 224],
  "theta_deg": float tensor [36],
  "emb": float16 tensor [V, 36, 2, 768],   # when using precomputed embeddings (default)
  "label_fine": int (−1 if none), "label_group": int (−1 if none),
  "weight": float,
  "prop_targets": {optic_class, pleo_cls, biref_cls, ext_type, twin_type} ints (−1 if undefined),
  "specimen": str, "candidates": multi-hot [n_fine_classes]
}
```

### 13.4 Splits (`data/processed/splits/folds.json`)
- Use 5 folds of **specimens** with `sklearn.model_selection.StratifiedGroupKFold` (groups = code, strata = rock_family).
- For each fold k: test = fold k. Validation = 15% of the remaining specimens (grouped, random seed 0). Train = the rest.
- Monomineralic specimens are split like any other specimen.
- Write the folds once and never regenerate them (seeded). All experiments use the same folds.

### 13.5 Class imbalance bookkeeping
Store per-fold training class counts `n_c` (fine and group). These are used for class weights and logit adjustment (§15.2).

---

## 14. Phase 8 — The model (RotaPetro-Net)

Module `model.py`. One main model; all baselines are switches of the same code.

### 14.1 Overview
```
for each angle k (36):          PPL crop_k ─┐                         ┌─ mineral head  (C_fine classes)
                                             ├─ frozen DINOv2-S ─► e_k ─► angle-aware  ─┼─ group head   (C_group)
                                XPL crop_k ─┘   (+mask pooling)      Transformer(2 layers)└─ property heads (explanations)
```

### 14.2 Frozen image encoder and embedding extraction (`embed.py`, script `08_embed.py`)
- **Encoder:** DINOv2 ViT-S/14 (`dinov2_vits14`, 21M parameters, 384-dim tokens). Frozen, in eval mode, fp16 on GPU.
- **Input:** 224×224 crop, ImageNet mean/std normalisation. This gives 16×16 = 256 patch tokens plus a CLS token.
- **Per crop output (768-d):** concatenate
  1. the CLS token (384), and
  2. the **mask-weighted mean of the patch tokens** (384). Downsample the region mask to 16×16 by area averaging and use it as the weights, so the vector describes the grain, not its neighbours.
- **Geometric versions (for augmentation, §15.4).** Compute embeddings for V = 4 versions of every region:
  - v0 = original
  - v1 = all crops rotated 90° + angle tags shifted by 90°
  - v2 = horizontal mirror + angle tags reversed
  - v3 = mirror + 90° rotation

  (8 versions are optional if disk allows.)
- **Storage:** `data/processed/embeddings/dinov2_vits14/emb.npy`, a float16 memmap `[N_regions, V, 36, 2, 768]`, plus `index.parquet` (region_id → row). About 0.44 MB per region for V=4, i.e. ≈ 2.2 GB for 5,000 regions.
- **Checks:** embeddings are deterministic (re-run gives identical values); no NaN.

### 14.3 Sequence model (trainable part, ≈ 1.5M parameters)
Inputs: `e_ppl[k], e_xpl[k] ∈ R^768` and `θ_k`.

1. **Token per angle:** `t_k = LayerNorm( W_p·e_ppl[k] + W_x·e_xpl[k] ) + AngleEmb(θ_k)`, with `W_p, W_x: 768 → 256`.
2. **AngleEmb:** an MLP (4 → 64 → 256, GELU) applied to `[sin 2θ, cos 2θ, sin 4θ, cos 4θ]`. These harmonics match the physics: PPL has a 180° period, XPL a 90° period.
3. **Sequence:** prepend a learnable `[CLS]` token to the kept angles (all 36, or a subset).
4. **Encoder:** `nn.TransformerEncoder` with 2 layers, d_model = 256, 4 heads, FFN 512, dropout 0.1, pre-norm, GELU, no positional encoding other than AngleEmb. The angle set is permutation-invariant by design, so fewer or irregular angles work.
5. **Pooled vector:** `h = concat(CLS_out, mean_k(t_k_out))` → Linear(512 → 256) → GELU → dropout 0.2.

### 14.4 Heads
- **Mineral head:** Linear(256 → C_fine).
- **Group head:** Linear(256 → C_group).
- **Property heads** (each Linear(256 → n)):

  | head | n | classes |
  |---|---|---|
  | optic_class | 3 | opaque / isotropic / anisotropic |
  | pleo_cls | 4 | none / weak / moderate / strong |
  | biref_cls | 5 | C0–C4 |
  | ext_type | 3 | parallel / inclined / undefined |
  | twin_type | 5 | none / simple / lamellar / crosshatched / sector |

  Property targets come from the Phase 4 features (automatic). At inference the predicted properties form the explanation.

### 14.5 Model variants (all in the same class, selected by config)

| id | name | what changes | purpose |
|---|---|---|---|
| **M1** | RotaPetro-Net (main) | as above, 36 angles, images only | main result |
| M2 | + physics | concatenate an MLP(features → 128) of the Phase 4 property card to h | hybrid (report carefully, §16.6) |
| M3 | + self-training | M1 retrained with pseudo-labels (§15.6) | final model |
| A-k | angles ablation | keep only k ∈ {1, 2, 4, 6, 9, 18, 36} evenly spaced angles (offset 0°) at train and test | E2 |
| A-ppl / A-xpl | modality ablation | zero the XPL or PPL embeddings | E3 |

### 14.6 Optional variant V3 (cross-attention)
Instead of adding PPL and XPL per angle, use two token streams and one `polar_vit_arch.Decoder` layer (from the LITHOS code) in which PPL tokens attend to XPL tokens and vice versa. Try it only after M1 works.

### 14.7 Baselines

| id | name | description |
|---|---|---|
| **B0** | Rule engine | §12.2 posterior, no training; constrained (C_s) and unconstrained versions |
| B1 | Physics + LightGBM | LightGBM on the Phase 4 features (defaults: 400 trees, lr 0.05, num_leaves 31, class_weight balanced) |
| **B2** | Static 0° (LITHOS-like) | M1 with only the 0° angle (T = 1): same encoder, same labels, same splits. **The key comparison.** |
| B3 (optional) | LITHOS PolarViT on 0° crops | the original architecture fine-tuned on our 0° PPL/XPL crops (heavy; batch 8, fp16) |

---

## 15. Phase 9 — Training procedure

Module `train.py`. Script `09_train.py --config configs/experiments/<exp>.yaml --fold k --seed s`.

### 15.1 Losses
`L = L_fine + 0.5·L_group + 0.2·Σ L_prop + λ_MIL·L_MIL`

- **L_fine:** weighted cross-entropy on regions with `label_fine`.
  - Label smoothing 0.1.
  - **Logit adjustment:** add `τ·log π_c` to the logits during training only, with π_c the training class frequencies and τ = 1.
  - Per-sample weight = `label_conf`. Also multiply by 1.5 for `mono` labels.
- **Hierarchical partial credit:** for regions with only `label_group = G`, use `L = −log Σ_{c ∈ G} p(c|x)`, added to L_fine.
- **L_group:** cross-entropy on the group head for all regions with a group (fine labels imply their group).
- **L_prop:** cross-entropy per property head, ignoring undefined targets (−1).
- **L_MIL (weak, bag-level):**
  1. Build a bag B of up to 16 regions from one specimen, labelled or not.
  2. The bag probability for class c is noisy-OR: `P_c = 1 − Π_{r ∈ B} (1 − p_{r,c})`.
  3. `L_MIL = −Σ_{c ∈ C_s} log P_c − β·Σ_{c ∉ C_s} log(1 − P_c)`, with β = 0.2. Negatives are weak because OU lists can omit minor minerals, and one site may not contain every listed mineral.
  4. Use λ_MIL = 0.3.

### 15.2 Sampling and class balance
- **Batch:** 96 labelled regions sampled with probability ∝ `n_c^(−0.5)` (square-root inverse frequency), plus 4 specimen bags × 16 regions for MIL.
- **Epoch:** 200 batches (not tied to dataset size).

### 15.3 Optimiser and schedule
- AdamW, lr 5e-4, weight decay 0.05, betas (0.9, 0.999).
- 3 warm-up epochs, then cosine decay to lr·0.01; at most 80 epochs.
- Early stopping on the validation metric, patience 15.
- **Validation metric** = mean of:
  - (a) macro-F1 on validation regions with fine labels, and
  - (b) mineral-set F1 on validation specimens (§16.2).
- Mixed precision; gradient clipping at 1.0; seeds 0, 1, 2.

### 15.4 Physics-consistent augmentation (important)
After alignment, the sample is fixed and the polariser turns relative to it by −rot_sign·θ. A geometric augmentation is valid **only** if it keeps the angle between polariser and crystal unchanged, which means rotating or mirroring the images **and** changing the angle tags together:

| augmentation | images | angle tags | valid? |
|---|---|---|---|
| rotate 90° | rotate all crops 90° | θ → θ + 90° (mod 180) | yes (precomputed version v1) |
| rotate 180° | rotate 180° | unchanged (180° periodicity) | yes |
| mirror | flip horizontally | θ → −θ (mod 180) | yes, if the polarisers are E–W/N–S (v2) |
| arbitrary rotation φ | rotate by φ | θ → θ + φ | valid physically, but needs re-embedding; skip in v1 |
| rotate images only (LITHOS style) | rotate | unchanged | **NO — never** |
| **angle dropout** | — | keep a random evenly spaced subset of K ∈ {6, 9, 12, 18, 36} angles with random offset (p = 0.5) | yes; makes the model work with fewer angles (videos) |
| angle jitter | — | add N(0, 1°) to θ | yes |
| colour jitter | brightness ±10%, per-channel gain ±5% (same for all angles, PPL and XPL) | — | yes, but needs re-embedding: use only in fine-tuning mode |
| hue shift | — | — | **NO** (hue encodes retardation) |

**Default training** (precomputed embeddings): pick a random version v ∈ {0..3} for each sample, then apply angle dropout and jitter.

### 15.5 Optional fine-tuning mode (only if M1 plateaus and a GPU with ≥ 16 GB is available)
- Unfreeze the last 2 DINOv2 blocks, lr 1e-5; compute embeddings on the fly with colour jitter.
- Batch 16 regions × 36 angles × 2; use gradient accumulation.

### 15.6 Self-training (M3)
After M1 converges on fold k:
1. Predict every **unlabelled** training region (all 4 geometric versions; average the probabilities).
2. Accept a pseudo-label if:
   - p ≥ 0.90, and
   - the class ∈ C_s, and
   - the class is in the rule engine's top 2 for that region.

   Cap new labels per class at max(50, current count).
3. Retrain from scratch with the pseudo-labels at weight 0.7.
4. Repeat for at most 2 rounds; stop when the validation metric does not improve by ≥ 0.005.

Never pseudo-label validation or test specimens.

### 15.7 What gets saved
`outputs/models/<exp>/fold<k>/seed<s>/`:
- `best.pt` and `last.pt` (LITHOS `save_epoch` pattern)
- `config.yaml` (full resolved config)
- `train_log.csv` (epoch, losses, lr, val metrics)
- `metrics.json` (final test metrics, §16)
- `predictions.parquet` (per test region: probabilities, predicted class, property predictions)

---

## 16. Phase 10 — Evaluation without human labels

Module `evaluate.py`. Script `10_evaluate.py --exp <exp>`. It aggregates over the 5 folds × 3 seeds and writes `outputs/reports/<exp>/`.

### 16.1 Test sets

| id | source | size (expected) | ground truth quality |
|---|---|---|---|
| **T1** | every test-fold specimen vs its OU mineral list | 115 specimens (all, across folds) | independent of the rules; set-level only |
| **T2** | grains of monomineralic specimens (§12.4) | depends on the collection; check Phase 0 | independent of the rules; grain-level |
| **T3** | centre grains of sites whose title names one mineral (§12.5) | depends on titles | independent of the rules; grain-level |
| T4 | rule-labelled test regions (§12.3) | largest | **not** independent: only for sanity checks |

### 16.2 Metrics
- **T1 mineral-set recovery.** For a test specimen, predict the set of minerals present:
  - a class c is present if the regions predicted as c cover ≥ 3% of the grain area of all its sites, or ≥ 2 regions have p(c) ≥ 0.6.
  - Map OU names through aliases. An OU group name matches any predicted member; accessory minerals are evaluated separately.

  Report precision, recall, F1 and Jaccard per specimen, then the macro average, in **two modes**: *unconstrained* (the model sees no candidate list) and *hinted* (predictions restricted to C_s).
- **T2 / T3 grain-level:** accuracy, macro-F1 over the classes present, per-class recall, confusion matrix. Also top-2 accuracy.
- **Calibration:** expected calibration error (15 bins) on T2 ∪ T3.
- **Properties:** accuracy of every property head against the Phase 4 property targets on test regions (explanation fidelity).
- **Coverage:** fraction of test regions predicted with p ≥ 0.6. Report the accuracy–coverage curve on T2 ∪ T3 (selective prediction, abstention).

### 16.3 Experiments

| id | question | comparison | primary metric |
|---|---|---|---|
| **E1** | Does rotation help? | M1 vs B2 (0° only) | T1 F1, T2/T3 accuracy |
| **E2** | How many angles are needed? | A-1, A-2, A-4, A-6, A-9, A-18, A-36 | same, plotted against angles |
| E3 | PPL or XPL rotation? | A-ppl vs A-xpl vs M1 | same |
| E4 | Physics vs learned features | B0 vs B1 vs M1 vs M2 | T1, T2, T3 |
| E5 | Does the network generalise beyond the rules it learned from? | M1/M3 vs B0 | T2/T3 accuracy (B0 is the teacher) |
| E6 | Value of weak (MIL) supervision and self-training | M1 without L_MIL, M1, M3 | T1, T2 |
| E7 | Robustness to video-like inputs | M1 on degraded test data (§16.5) | T1, T2 |

### 16.4 Statistics
- Report mean ± std over 5 folds × 3 seeds.
- For E1 and E2, use a **paired bootstrap over test specimens** (10,000 resamples) of the difference M1 − B2, and report the 95% CI. The claim "rotation helps" is made only if the CI excludes 0.

### 16.5 Robustness tests (simulated video conditions, applied to the test stacks)
- JPEG re-compression at quality 50;
- Gaussian blur σ = 1.5 px;
- white-balance gain ±10% per channel;
- brightness ±15%;
- angle jitter N(0, 2°);
- fewer angles (6, 9, 12);
- random 10% of angles missing.

### 16.6 Honesty rules for reporting
- **Never** report T4 (rule-label agreement) as accuracy. It measures agreement with the teacher, and it is biased in favour of models that use the same physics (M2, B1).
- Every number on T2/T3 must list how many grains and how many specimens it is based on.
- Report classes with fewer than 10 test grains, but flag them as unreliable.

---

## 17. Phase 11 — Inference: from a rotation video to a mineral report

Module `infer_video.py`. Script `12_infer.py --ppl ppl.mp4 --xpl xpl.mp4 --meta meta.json --out report/`.

### 17.1 Capture protocol for users (put this in the README and the app)
1. Standard petrographic microscope; Köhler illumination; polarisers E–W / N–S (standard); objective 4× or 5× (field ≈ 2–4 mm; the target is ≈ 4 µm/px after resizing).
2. Centre the stage (a point at the crosshair must not wander during rotation).
3. **Camera: lock exposure, gain, white balance and focus.** Turn off every "auto" setting. Record ≥ 1080p at ≥ 24 fps with the highest-quality codec available.
4. **Pass 1 (PPL):** analyser out. Rotate the stage slowly and steadily through **at least 180° (360° preferred)** in about 20–40 s.
5. **Pass 2 (XPL):** analyser in, same field, same settings. Rotate again through ≥ 180°.
6. **Metadata file:** objective, camera, µm/px if known (or one photo of a stage micrometer), sample name, and (optional) candidate minerals or rock type.

### 17.2 Pipeline

| step | what happens | how | failure → action |
|---|---|---|---|
| 1 | Decode frames | imageio/OpenCV, RGB | — |
| 2 | Quality checks | blur: variance of Laplacian ≥ threshold; clipping < 5% of pixels at 255; background brightness drift < 5% across the pass (detects auto-exposure) | warn or reject |
| 3 | Field detection | Hough circle on the dark border → centre, radius; crop a square | fall back to the full frame |
| 4 | Stage angle per frame | ORB + RANSAC similarity to the first frame; for angles > 60°, chain through keyframes every ~20° and accumulate; unwrap | need ≥ 180° coverage, else reject |
| 5 | Pick angles | for θ_k = 0, 5, …, 175° choose the frame nearest θ_k (+180° frames averaged in if available); max gap 3° | missing angles allowed (model is angle-robust); require ≥ 6 |
| 6 | Align | warp each chosen frame back to the reference orientation (as in §7) | — |
| 7 | PPL↔XPL co-registration | register the XPL max-projection to the PPL mean on gradient images (ECC, Euclidean) → common coordinates and angle offset between passes; shift the XPL angle tags | reject if the match is poor |
| 8 | Scale | resize so that µm/px = `um_per_px_rot` of the training data | warn if the scale is unknown |
| 9 | Colour | white balance on clear PPL areas (§8.4); normalise the white level | — |
| 10 | Maps, segmentation, features | Phases 2–4 unchanged | — |
| 11 | Embeddings + model | DINOv2-S on the 36 (or fewer) angles; M3 (or M1) weights; average over the 4 geometric versions (test-time augmentation) | — |
| 12 | Rule engine (unconstrained, or hinted if candidates are given) | computed in parallel | — |
| 13 | Decision | final label = model argmax. If p < 0.6, or the model's top-1 is not in the rule top-3 → **"uncertain"** (give the top 2) | — |
| 14 | Report | JSON + overlay PNG + text | — |

### 17.3 Output JSON (`report/field.json`)
```json
{
  "field_id": "user_sample_A_f1",
  "qc": {"angles_used": 34, "coverage_deg": 360, "blur_ok": true, "exposure_locked": true, "scale_um_per_px": 4.0},
  "grains": [
    {"id": 7, "bbox_xywh": [120, 80, 64, 40], "area_um2": 31200,
     "prediction": {"label": "biotite", "group": "mica", "confidence": 0.91,
                    "top3": [["biotite", 0.91], ["hornblende", 0.05], ["tourmaline", 0.02]]},
     "properties": {"optic_class": "anisotropic", "ppl_colour": "brown", "pleochroism": "strong",
                    "biref_class": "C3 (masked by colour)", "extinction": "parallel (2°)",
                    "twinning": "none", "cleavage_dirs": 1, "absorption_vs_long_deg": 4},
     "explanation": "Brown and strongly pleochroic; darkest when its cleavage is parallel to the polariser; parallel extinction; one cleavage. Hornblende is less likely: no second cleavage and extinction is not inclined. Tourmaline is less likely: its absorption axis would be perpendicular to the elongation."}
  ],
  "summary": {"minerals": [{"label": "quartz", "area_pct": 41.2, "n_grains": 9, "mean_conf": 0.88}],
              "uncertain_pct": 7.5, "fine_aggregate_pct": 3.1,
              "assemblage_hint": "quartz + K-feldspar + plagioclase + biotite → granitoid (hint only; one field is not the whole rock)"}
}
```

### 17.4 Explanation generator (`explain.py`)
Templates are filled from:
- the **predicted property heads** ("strongly pleochroic", "parallel extinction");
- the **measured features** (numbers such as the extinction angle, the cleavage count);
- the knowledge-base `special` text;
- the **rule-engine evidence** for rejecting the runner-up: the feature with the most negative contribution for mineral #2.

Never generate facts that are not in these four sources.

### 17.5 Hint mode
If the user supplies candidate minerals or a rock type, restrict the posterior to those candidates by renormalising the probabilities over C_user. This is the same mechanism as the "hinted" evaluation mode (T1).

### 17.6 Assemblage hints (`knowledge/assemblage_hints.yaml`)
Simple, transparent rules applied to the field (or multi-field) summary. Examples:
- quartz + K-feldspar + plagioclase (+ biotite/muscovite) → granitoid
- plagioclase + augite (± olivine, opaque) → basalt / dolerite / gabbro family
- olivine ≫ others → peridotite / dunite
- calcite or dolomite ≫ others → limestone / marble
- quartz ≫ others → quartzite / quartz arenite
- hornblende + plagioclase → amphibolite / diorite
- muscovite/biotite + quartz + garnet/staurolite/kyanite/sillimanite → pelitic schist or gneiss

Always label it "hint". A proper rock name needs modal composition over many fields and textures.

### 17.7 Performance targets
- Laptop CPU: < 3 min per field (embeddings dominate).
- GPU: < 30 s.

---

## 18. Expected results, success criteria and the paper

### 18.1 What we expect (hypotheses, not promises)
- **Rule engine (B0):** strong on minerals with unmistakable rotation signatures: biotite, muscovite, garnet, opaque, calcite/dolomite, twinned plagioclase, microcline, olivine. Weak on untwinned feldspar vs quartz, orthopyroxene vs augite and some amphiboles.
- **RotaPetro-Net (M1/M3):** at least as good as B0 on T2/T3, and better on grains where the rules lack evidence (small, partial or fractured grains), because it also uses texture, cleavage and zoning.
- **Rotation vs 0° (E1):** the largest gains are expected on twinned feldspars, micas vs pleochroic look-alikes, isotropic minerals, and extinction-angle-dependent pairs.
- **Angles (E2):** we expect most of the gain to appear by 6–9 angles (every 20–30°), with little improvement beyond 18. This curve is the practical recommendation for future datasets.
- **For context only:** LITHOS reported 25-class macro-F1 ≈ 0.43 with static 0° images on a different dataset, so the numbers are not directly comparable.

### 18.2 Success criteria (decide before running the experiments)
- **S1.** M1 beats B2 on T1 mineral-set F1 (unconstrained) with a 95% bootstrap CI that excludes 0.
- **S2.** M1 or M3 reaches ≥ 0.85 accuracy on T2 (monomineralic grains), if T2 contains ≥ 3 specimens of ≥ 2 minerals.
- **S3.** Property heads reach ≥ 0.8 accuracy for optic_class, pleo_cls and biref_cls (the explanations are faithful).
- **S4.** The E2 curve is monotone, with a clear knee.
- **S5.** In robustness tests (§16.5), T1 F1 drops by less than 0.10 with 9 angles plus JPEG and blur.

If a criterion fails, that is still a result. Report it honestly: for example, "rotation helps only for X classes".

### 18.3 Results tables to fill (templates)
**Table A — main comparison (mean ± std over folds × seeds):**

| model | T1 F1 (uncon.) | T1 F1 (hinted) | T2 acc | T2 macro-F1 | T3 acc | ECE |
|---|---|---|---|---|---|---|
| B0 rules | | | | | | |
| B1 physics+GBM | | | | | | |
| B2 0° only | | | | | | |
| M1 rotation | | | | | | |
| M2 rotation+physics | | | | | | |
| M3 rotation+self-training | | | | | | |

**Table B — angles ablation:** rows k = 1, 2, 4, 6, 9, 18, 36; the same columns.

**Table C — per-class recall on T2/T3, M3 vs B2,** with n grains and n specimens per class.

### 18.4 Paper plan
- **Working title:** "Does stage rotation matter? Label-free mineral identification from polarised-light rotation series."
- **Contributions:**
  1. A rotation-series grain dataset derived from UKVM, with automatic, physics-based labels and zero manual annotation.
  2. A physics-grounded rule engine and an angle-aware network.
  3. A quantified gain of rotation over 0° imaging, and the number of angles needed.
  4. Explainable predictions.
  5. An open inference pipeline and a capture protocol for rotation videos.
- **Figures:**
  - pipeline;
  - property maps of two sites;
  - segmentation examples;
  - an E2 curve (accuracy vs number of angles);
  - confusion matrices for B2 vs M3;
  - example explanations;
  - a robustness plot.
- **Venues:** *Computers & Geosciences*; *Artificial Intelligence in Geosciences*; an EGU/AGU abstract first.
- **Release:** code (MIT or Apache-2.0), derived labels and maps (CC BY-NC-SA 2.0, as for the OU images), and scripts to re-download the OU images from the source with attribution.

---

## 19. Limitations and risks

### 19.1 Limitations (state them in the paper)
1. **Small data:** 115 specimens, one collection, a few thousand usable grains. Rare minerals merge into groups or "other".
2. **One acquisition system:** OU's microscope and camera, JPEG, ≈ 4 µm/px. Transfer to other microscopes or phones is untested until we collect our own data.
3. **Non-random sites:** OU chose "interesting" places, so the mineral proportions in sites are not the rock's modal composition.
4. **Unmeasurable properties:** relief is only a proxy (no Becke line), and there is no optic sign, 2V, λ-plate or reflected light. Some pairs remain ambiguous: quartz vs untwinned orthoclase/cordierite, calcite vs dolomite, some amphiboles and pyroxenes.
5. **Automatic labels are noisy,** and their systematic errors are not measured grain by grain. T2 and T3 cover only some minerals.
6. **Segmentation errors** (merged or split grains) are not corrected by humans.
7. **Fine-grained material** is only "fine_aggregate".
8. **A single field is not a rock:** rock names are hints only.

### 19.2 Risks and mitigations

| risk | likelihood | mitigation |
|---|---|---|
| Some sites fail to align | low | fallback to the nominal 5°·i rotation; QC; drop bad sites |
| Segmentation over- or under-merges | medium | tune on 20 sites; twin rule based on PPL boundary strength; optional SAM |
| Rule engine mislabels a whole class | medium | per-class contact sheets; likelihood tuning; group-level fallback; T2/T3 checks |
| Too few labelled grains per class | high | group classes; MIL; self-training; report per-class support |
| OU lists incomplete or generic | high | aliases; group expansion; weak negatives (β = 0.2); always-allowed opaque/aggregate |
| Few monomineralic specimens or no mineral titles | medium | T1 still works; report T2/T3 sizes honestly |
| Duplicate sites leak across splits | high if not fixed | Phase 0 dedupe; splits by specimen |
| Video inference fails (auto-exposure, bad centring) | high | protocol; QC rejects; clear error messages |

---

## 20. Timeline and definition of done

A team of 3–4, part-time. Scripts are numbered in run order.

| week | phase / script | done when |
|---|---|---|
| 1 | `00_clean.py`, aliases, scale check | `phase0_report.md` reviewed; `sites.csv` free of duplicates; pixel size fixed in config |
| 1–2 | `01_align.py` | all usable sites aligned; QC pass criteria (§7.3) met on ≥ 95% |
| 2 | `02_maps.py`, `michel_levy.py` | LUT sanity colours correct; maps visually correct on granite and 11agt25 |
| 3 | `03_segment.py` | 20-site overlay review done; parameters frozen |
| 3–4 | `04_features.py`, knowledge base v1 | `regions.parquet` complete; sign/polariser calibration done (§10.6); `minerals.csv` verified from two sources |
| 4–5 | `06_autolabel.py` | coverage 40–70%; per-class contact sheets plausible; `classes.yaml` written |
| 5 | `07_build_dataset.py`, `08_embed.py` | folds frozen; embeddings stored and verified |
| 6–7 | `09_train.py` B0, B1, B2, M1 | 5 folds × 3 seeds finished |
| 8 | ablations E2, E3; M2; M3 self-training | all experiment tables filled |
| 9 | `10_evaluate.py`, robustness | report folder complete |
| 10 | `12_infer.py` + one real test video recorded at IIT Kanpur | report produced for the video |
| 11–12 | paper writing, release | draft submitted to the supervisor |

---

## 21. Common mistakes to avoid

1. **Splitting by region or site instead of specimen.** This leaks information and inflates results.
2. **Leaving duplicate sites** (r3 ≡ r1). This double-counts data and leaks.
3. **Using LITHOS-style `RandomRotation`,** or any image rotation without shifting the angle tags. It is physically wrong.
4. **Hue jitter.** Interference colours encode retardation.
5. **Re-saving frames as JPEG.** Every step after download must be lossless (PNG/NPZ).
6. **Averaging frame i with frame i+36 before alignment.** Always align first.
7. **Treating OU generic names** ("feldspar", "mica") as specific minerals. Expand them to groups.
8. **Reporting agreement with rule labels (T4) as accuracy,** especially for models that see physics features (M2, B1).
9. **Treating OU lists as complete** (hard negatives in MIL). Use weak negatives.
10. **Mixing angle conventions:** degrees vs radians, 0–90 vs 0–180 periods, the image y-axis pointing down. Test with the §10.6 calibration.
11. **Using whole-slide pixels together with rotation-frame pixels** without scale conversion.
12. **Tuning thresholds on test-fold specimens.** Tune on the training folds of fold 1, then freeze.
13. **Forgetting `valid` masks.** Circle borders create fake edges and fake regions.
14. **Inference videos recorded with auto-exposure.** This breaks the extinction curves. Reject them in QC.

---

## 22. Appendices

### 22.A `configs/default.yaml` (all tunable values in one place)
```yaml
paths:
  raw: data/raw/ukvm_dataset
  interim: data/interim
  processed: data/processed
  knowledge: knowledge
  outputs: outputs
acquisition:
  n_raw_frames: 72
  step_deg: 5.0
  frame_size: 490
  valid_radius_px: 232
  um_per_px_rot: 4.0            # confirm in Phase 0 (§6.6)
  rot_sign: null                # set by §7.1 / §10.6 calibration
  polariser_deg: 0.0            # E-W; confirm in §10.6
align:
  orb_features: 3000
  n_matches: 600
  ransac_px: 2.0
  max_angle_err_deg: 1.5
  max_scale_err: 0.01
maps:
  aniso_min: 0.15
  high_order_white: {dE: 25, L: 75, chroma: 15, relief_mod: 0.3}
  biref_class_nm: [100, 350, 1100, 2000]
  pleo_class_dE: [4, 10, 20]
  opaque_rel_L: 0.15
segment:
  slic_px_per_segment: 150
  slic_compactness: 10
  T_merge: 14
  twin_merge: {dE_ppl: 6, max_biref_diff: 1, boundary_grad_ratio: 0.5, min_shared_px: 15}
  min_region_px: 150
  min_sample_px: 400
  hole_rel_L: 0.92
  aggregate: {circvar: 0.6, r2_4: 0.4}
features:
  border_exclude_px: 2
  min_aspect_for_axis: 1.3
  twin: {min_frac: 0.10, min_sep_deg: 8}
  cleavage: {sato_sigmas: [1, 2], ridge_pct: 95, hough_min_len: 12, peak_frac: 0.20}
rules:
  accept_p: 0.80
  accept_margin: 0.40
  group_p: 0.85
  min_r2_4: 0.4
  min_features: 4
  always_allowed: [opaque, fine_aggregate]
classes:
  min_regions: 30
  min_specimens: 3
dataset:
  crop_scale: 1.3
  crop_min_px: 48
  input_px: 224
  context_dim: 0.35
  folds: 5
  val_frac: 0.15
  seed: 0
model:
  encoder: dinov2_vits14
  n_versions: 4
  d_model: 256
  n_layers: 2
  n_heads: 4
  ffn: 512
  dropout: 0.1
train:
  lr: 5.0e-4
  weight_decay: 0.05
  epochs: 80
  warmup_epochs: 3
  patience: 15
  batch_labelled: 96
  bags_per_batch: 4
  bag_size: 16
  label_smoothing: 0.1
  logit_adjust_tau: 1.0
  lambda_group: 0.5
  lambda_prop: 0.2
  lambda_mil: 0.3
  mil_neg_beta: 0.2
  mono_weight: 1.5
  angle_dropout: {p: 0.5, K: [6, 9, 12, 18, 36]}
  angle_jitter_deg: 1.0
  seeds: [0, 1, 2]
selftrain:
  p_min: 0.90
  pseudo_weight: 0.7
  rounds: 2
infer:
  min_angles: 6
  max_angle_gap_deg: 3
  uncertain_p: 0.6
```

### 22.B Formula quick reference
- **XPL model:** `I(θ) = I_bg + A·sin²(2(θ−θ₀)) = I_bg + A/2 − (A/2)·cos(4(θ−θ₀))`
- **4th harmonic:** `c4 = (2/N)ΣL_k cos 4θ_k`, `s4 = (2/N)ΣL_k sin 4θ_k`, `θ₀ = ((atan2(s4,c4) + π)/4) mod (π/2)`
- **Anisotropy:** `A4/(m0+1)`. **Fit quality:** `(A4²/2)/var(L)`
- **PPL pleochroism:** `I_c(θ) = a_c + b_c·cos(2(θ−θ₁))` per channel; ΔE76 between the extremes
- **Retardation:** `Γ = d·δ`, with d ≈ 30 µm, so Γ(nm) ≈ 30 000·δ
- **Michel-Lévy colour:** `S(λ) = D65(λ)·sin²(πΓ/λ)` → XYZ (CIE 1931 2°) → sRGB/Lab
- **Circular mean with period 90°:** `z = Σ w·e^{i4θ}`; mean = arg(z)/4; `R = |z|/Σw`; circular std = `sqrt(−2 ln R)/4`
- **Noisy-OR bag probability:** `P_c = 1 − Π(1 − p_{r,c})`
- **Logit adjustment:** `logits_train = logits + τ·log π_c`

### 22.C Glossary
- **PPL / XPL:** plane-polarised / cross-polarised light.
- **Extinction:** a grain goes dark in XPL when its vibration direction is parallel to the polariser; this happens every 90°.
- **Retardation (Γ):** thickness × birefringence; it sets the interference colour.
- **Pleochroism:** a colour change in PPL on rotation.
- **Site:** one rotation field (≈ 2 mm circle).
- **Region:** a segmented grain or aggregate.
- **Rule engine:** the knowledge-based scorer that creates automatic labels.
- **MIL:** multiple-instance learning; training from bag-level (specimen) labels.
- **Self-training:** retraining with the model's own confident predictions.
- **Logit adjustment:** a correction for class imbalance.
- **ECE:** expected calibration error.

### 22.D Existing tools (in `tools/`)
- **`ukvm_download.py`:** downloads the 115 UKVM specimens. Known issues are listed in §5; Phase 0 fixes them. For future downloads, patch it to (a) deduplicate rotation plans by `(pplData, xplData)`, (b) uppercase whole-slide labels, and (c) write `accessory_minerals` correctly.
- **`physics_maps.py`:** the working prototype of alignment (ORB + RANSAC) and of the extinction, anisotropy and pleochroism maps. It was tested on the granite (4 sites) and on 11agt25. Use it as the reference implementation for §7–§8, then refactor into `align.py` and `maps.py` with the folding, white balance and Michel-Lévy additions.

### 22.E Reading list for the team (minimum)
1. Nesse — *Introduction to Optical Mineralogy*: chapters on polarised light, the indicatrix and interference colours, plus the mineral descriptions for the classes above.
2. Raith, Raase & Reinhardt — *Guide to Thin Section Microscopy*: the optical property tables.
3. Oquab et al. 2023 — DINOv2 (the frozen encoder).
4. Vaswani et al. 2017 — the Transformer (the sequence model).
5. Ilse et al. 2018 — attention-based multiple-instance learning (background for MIL).
6. Menon et al. 2021 — long-tail learning via logit adjustment.
7. Chacón et al. 2025 — LITHOS (the static baseline and dataset we compare against).
