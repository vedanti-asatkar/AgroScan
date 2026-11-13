# Architecture

```mermaid
flowchart LR
  subgraph Data["Data (DVC)"]
    PV[PlantVillage] --> PREP
    PD[PlantDoc] --> PREP
  end
  subgraph Pipeline["DVC pipeline (Person A)"]
    PREP[prepare] --> VAL[validate] --> TRAIN[train] --> EVAL[evaluate] --> EXP[export] --> DRIFT[drift]
    TRAIN -. runs/params .-> MLF[(MLflow)]
  end
  EXP --> META[models/model_meta.json<br/>+ model + classes.json]
  META --> API[FastAPI + UI]
  API --> LOG[(logs/predictions.jsonl)]
  META --> BATCH[scripts/batch_predict.py]
  LOG --> MON[src/monitor_logs.py]
  DRIFT --> MON
  EVAL --> MON
  CFG[configs/monitoring.yaml] --> MON
  MON --> SUM[reports/monitoring/summary.json]
  SUM --> BR{Airflow<br/>check_thresholds}
  BR -- healthy --> HR[healthy_report.json]
  BR -- retrain --> RD[retrain_decision.json] --> TRAIN
  GIT[merge to main] --> REL[release.yml] --> GHCR[(ghcr.io image)]
  GHCR --> RUN[docker run]
```

## Automation layers

| Layer | File | Trigger | Does |
|---|---|---|---|
| CI (A) | `.github/workflows/ci.yml` | push / PR | lint, tests, synthetic smoke test, Docker build |
| Delivery (B) | `.github/workflows/release.yml` | merge to `main` | build image, push to GHCR, smoke-test `/health` |
| Scheduled pipeline (B) | `airflow/dags/agroscan_pipeline.py` | weekly | validate, run pipeline, monitor logs, branch healthy / retrain |

## The serving contract

The API, batch script and notebooks all read `models/model_meta.json` (`img_size`, `input_range`, `center_crop`, `confidence_threshold`, `top_k`, ...). Change preprocessing only through that file and `src/preprocess.py`; never hard-code it elsewhere.
