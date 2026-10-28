# Third-party sources and figure attribution

## Microscopy images

- Collection: The Open University Virtual Microscope for Earth Sciences.
- Collection site: https://www.virtualmicroscope.org/
- Running example: granite, specimen page https://virtualmicroscope.org/node/35 ; rotation site `granite_r1`.
- Retained specimen metadata licence text: **“CC BY-NC-SA 2.0 (Open University Virtual Microscope), unless marked third-party.”**
- Retained owner URL: http://www.open.ac.uk
- Full underlying specimen metadata is provided in `evidence/granite_source_metadata.json`.

The report images `raw.png`, `align.png`, `maps.png`, `segment.png`, `selected.png`, `crops1.png`, `crops2.png` and `prediction_maps.png` contain or derive from these source photographs. The project changes include layout, registration, angle-pair averaging, cropping, dimming context, optical maps, outlines, labels and prediction overlays. The originals were not captured by this project. Retain the original attribution, non-commercial and share-alike notice and any collection-specific exceptions when redistributing microscope-derived material. This notice records retained metadata, not a fresh legal determination of every collection asset's rights.

`optical_curves.png` plots measurements reconstructed from the same local image stack. `class_results.png`, `confusions.png`, `learning_curves.png` and `probabilities.png` plot project experiment evidence; `views_flow.png` is a project process diagram.

## Methods and pretrained model

DINOv2: https://github.com/facebookresearch/dinov2 and https://arxiv.org/abs/2304.07193 . The encoder/weights are downloaded separately; consult their own licence and model card.

ORB, RANSAC, ECC, SLIC and transformer references appear in the report bibliography. Python and TeX dependencies are governed by their respective licences and are not bundled in this repository.
