# AgroScan presentation outline (12 slides)

Lead with measured results. Keep each result paired with its report file and sample count; PlantDoc and PlantWild estimates come from different evaluation reports and cleaning policies.

1. **Title and measured result:** pipeline candidate 44.4% top-1 on 72 PlantDoc images (`reports/metrics.json`) versus v5 58.3% top-1 / 94.4% top-3 on the same reported test split (`reports/v5_plantdoc_metrics.json`). State that prepared archive reports also differ.
2. **Problem and intended scope:** single-leaf photo, eight pepper/potato/tomato classes, decision support rather than agronomist replacement.
3. **Architecture:** DVC graph, offline evaluation, serving API, prediction-log monitor, Airflow and GitHub Actions delivery (`docs/architecture.md`).
4. **Data and validation:** PlantVillage and PlantDoc training pool, held-out PlantDoc DVC test split, 10,240 / 1,808 / 72 image totals (`reports/data_validation.json`); explain separate prepared-archive duplicate audit.
5. **Reproducible ML lifecycle:** eight exact `dvc.yaml` stages, model metadata, candidate quality gate, MLflow tracking.
6. **Results across evaluations:** v5 and candidate top-1/top-3 with their exact reports and sample counts, including cleaned vs all-image differences.
7. **Per-class behavior and confusion matrix:** use `reports/v5_plantdoc_prepared_metrics.json` for the cleaned v5 table and label `reports/confusion_matrix.png` as pipeline candidate on the 72-image DVC test split.
8. **Serving:** FastAPI `/health`, `/ready`, `POST /predict`, UI at `/ui/`, preprocessing selected by model metadata.
9. **Monitoring and retraining:** offline `src.drift_check`, live `src.monitor_logs`, `configs/monitoring.yaml`, and export gate.
10. **Airflow automation:** weekly `agroscan_pipeline` and branch to the healthy or retrain report. Add a placeholder box labelled **SCREENSHOT** for the Airflow graph view.
11. **Responsible use and privacy:** scope, small test samples, confidence limitations, exact prediction-log fields, and explainability caveat. Add a placeholder box labelled **SCREENSHOT** for the UI.
12. **What comes next:** fix renderer/report shape incompatibility, confirm packaged model artifacts, reconcile evaluation versions, and improve real-world performance before broader deployment.

## Data sources for slides

| Claim | Report |
|---|---|
| Candidate current PlantDoc accuracy, validation metrics | `reports/metrics.json` |
| Candidate class metrics and confusion matrix | `reports/classification_report.json`; `reports/confusion_matrix.png` |
| v5 current 72-image PlantDoc evaluation | `reports/v5_plantdoc_metrics.json` |
| v5 archive-cleaned PlantDoc and PlantWild evaluations | `reports/v5_plantdoc_prepared_metrics.json`; `reports/v5_plantwild_metrics.json` |
| v5 PlantWild including duplicates | `reports/v5_plantwild_metrics_including_duplicates.json` |
| Candidate archive-cleaned PlantDoc and PlantWild evaluations | `reports/pipeline_plantdoc_prepared_metrics.json`; `reports/pipeline_plantwild_prepared_metrics.json` |
| Validation split counts | `reports/data_validation.json` |
| Cross-split duplicates and exclusions | `reports/prepared_dataset_audit.json` |
| Drift features | `reports/drift.json` |

The required command `python scripts/render_results_table.py` currently raises `KeyError: 'precision'` because `reports/classification_report.json` nests class tables under `val` and `test`. Until the renderer is corrected outside this documentation task, copy tables directly from the report JSON and identify their source on the slide.
