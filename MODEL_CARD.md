# Model card: AgroScan leaf disease classifier

> Fill every `TODO(measured)` from `reports/` once Person A's `dvc repro` results are committed
> (`python scripts/render_results_table.py` prints the tables). Until then, only the numbers marked
> *reported* below are available, and they have not been re-measured in this repo.

## Model details

- **Task:** image classification of a single leaf photo into 8 pepper / potato / tomato categories.
- **Architecture:** MobileNetV2 (transfer learning), Keras.
- **Served model (v5):** 288 px input, raw 0-255 pixel values, centre crop, confidence threshold 0.7, top-3 output. The pipeline-trained candidate uses 224 px and 0-1 values. The API follows `models/model_meta.json` for each model.
- **Version:** TODO(measured): MLflow run ID of the deployed model (Person A, task A5).
- **Owners:** TODO: team names. **Contact:** TODO.

## Intended use

- **In scope:** a learning / demonstration tool that suggests the likely disease of a pepper, potato or tomato leaf from a photo, with the top-3 candidates and a low-confidence warning.
- **Out of scope:** the only decision-maker for treatment, crop-loss or pesticide decisions; other crops or plant parts (fruit, stems); other diseases, pests or nutrient deficiencies; regulatory or insurance use. Always confirm with an agronomist or extension service.

## Data

| Source | Use | Notes |
|---|---|---|
| PlantVillage | training / validation | Lab-style images: single leaf, uniform background. |
| PlantDoc | **test only** | Field images with cluttered backgrounds. Never used for training or tuning. |
| PlantWild | extra test report | Cleaned test images. |

- Folder names are mapped to the 8 classes; exact duplicate images are removed within the training pool.
- A data audit found **46 exact-duplicate groups crossing splits**, some with conflicting labels. TODO(measured): record how A's validation / leakage guard handled them and re-state the final counts.
- TODO(measured): class counts per split from `reports/data_validation.json`.

## Performance

*Reported* (training notebook / earlier reports, **not** re-measured here; they also disagree with each other):

| Model | Test set | Top-1 | Top-3 | n |
|---|---|---|---|---|
| v5 | PlantDoc | 62.5% (README / notebook) vs 58.3% (committed report) | 94.4% | 72 |
| v5 | PlantWild | 60.9% (notebook, 304 imgs) vs 60.5% (committed report, 299 imgs) | 89.0% | 299-304 |
| Pipeline candidate | validation / PlantDoc | 88.4% / 44.4% | n/a | n/a |

**TODO(measured):** replace with the numbers from `reports/metrics.json` and add the per-class table below.

### Per-class results

TODO(measured): paste the output of `python scripts/render_results_table.py` (precision, recall, F1, support per class), and embed `reports/confusion_matrix.png`.

**How to read these numbers:** the test sets are tiny, so a difference of a few points is noise. The large gap between validation (about 88%) and PlantDoc (about 44-60%) is the real finding: **domain shift** from lab to field images.

## Explainability

`notebooks/lime_example.ipynb` and `notebooks/shap_example.ipynb` produce saliency examples in `reports/explainability/`. TODO(measured): after running them, note 2-3 findings (for example whether highlighted regions fall on lesions or on background).

## Limitations

- **Domain shift:** trained mostly on clean, single-leaf images; accuracy on field photos is much lower. Brightness and RGB statistics differ measurably between training and PlantDoc (committed drift report).
- **Only 8 classes:** any other plant or disease is forced into one of the 8. The model has no "none of these" class; use the confidence warning and your own judgement.
- **Low-confidence handling:** predictions under 0.7 are flagged, but high confidence is not a guarantee of correctness.
- **Similar-looking diseases** (for example early vs late blight) are the likely confusions; see the confusion matrix.
- **Small test sets;** results have wide uncertainty.
- Photo quality (blur, lighting, several leaves, hand-held shadows) degrades results.

## Privacy and data handling

- Uploaded images are processed in memory for prediction. TODO(confirm in `api/main.py`): the API does **not** store the uploaded image.
- The prediction log (`logs/predictions.jsonl`) stores only a summary: timestamp, model version, predicted class, confidence, top-3, latency and mean brightness. No image, filename, IP address or user identifier. TODO(confirm) that this stays true.
- Leaf photos can contain faces, hands, locations or EXIF metadata. Do not upload photos of people; do not add any image storage without a retention policy and consent.
- `logs/` is git-ignored; do not commit logs.

## Governance

- **Monitoring:** weekly `src.monitor_logs` summary; thresholds in `configs/monitoring.yaml`; policy in `docs/retraining_policy.md`.
- **Retrain / deploy:** a retrain is triggered by drift, accuracy below the minimum, or a high low-confidence rate, but a candidate is exported only if it passes the accuracy gate in `params.yaml`.
- **Change control:** all changes via pull request with the other team member reviewing; model and data versions via DVC and MLflow; image published by `release.yml`.
- **Data rules:** the PlantDoc test set is never used for training or edited. No secrets in code or notebooks.
- **Fairness:** not assessed. There are no demographic groups in the data; the relevant "subgroups" are crops, diseases and image sources, which the per-class table covers.
- **Review:** TODO: date of last model-card review.
