# Deploying AgriSignal to Google Cloud Run

AgriSignal runs on Google Cloud as one container image deployed three ways, with a
Cloud Storage bucket standing in for the local `data/` directory.

```
                 Cloud Scheduler (07:00 UTC, Mon–Fri)
                              │ POST …/jobs/agrisignal-pipeline:run
                              ▼
┌────────────────────────────────────────────┐
│ Cloud Run job  agrisignal-pipeline         │  ingest → silver → gold → quality → train
└─────────────────────┬──────────────────────┘
                      │ writes                  Secret Manager: config.yaml
                      ▼                         mounted at /secrets/config.yaml
┌────────────────────────────────────────────┐
│ GCS bucket  <project>-agrisignal-data      │  mounted at /app/data (Cloud Storage FUSE)
│ bronze/ silver/ gold/ models/ mlruns/      │  object versioning on
└──────────┬──────────────────────┬──────────┘
           │ reads                │ reads
           ▼                      ▼
┌──────────────────────┐  ┌─────────────────────────┐
│ agrisignal-api       │  │ agrisignal-mlflow       │
│ FastAPI, public      │  │ MLflow UI, private      │
└──────────────────────┘  └─────────────────────────┘
```

| Resource | What it is |
|---|---|
| `agrisignal-api` | Cloud Run service, public. Serves `/predict`, `/health`, `/model/metadata`. Scales to zero. |
| `agrisignal-pipeline` | Cloud Run job. Runs `agrisignal.orchestration.flows.daily_pipeline`; trains on Mondays. |
| `agrisignal-mlflow` | Cloud Run service, private (IAM). `mlflow server` over the same bucket. |
| `agrisignal-daily` | Cloud Scheduler job that executes the pipeline job. |
| `<project>-agrisignal-data` | GCS bucket mounted at `/app/data` in all three. |
| `agrisignal-config` | Secret Manager secret holding `configs/config.yaml` (includes the NOAA token). |
| `agrisignal` repository | Artifact Registry repository; images are tagged with the git commit. |

Region: `europe-west2` (London). The docker-compose extras (Postgres, Prefect server,
Prometheus, Grafana) are not deployed: Cloud Scheduler and Cloud Run jobs replace the
Prefect server for this single daily flow, which runs Prefect with a temporary local
server inside the job.

## Versioning and lineage

Every training run records what went into the model and what came out of it.

- **Model versions.** Each training registers a new version of the MLflow registered
  model `agrisignal-corn` and moves the `champion` alias to it. The API serves the
  latest trained model, so `champion` is always the version being served.
- **Dataset versions.** Each run logs the gold feature matrix as an MLflow dataset
  input: a content digest, schema, row count and source path, plus run tags
  `dataset.digest`, `dataset.rows`, `dataset.start_date` and `dataset.end_date`. The
  digest is also tagged on the model version. Identical data gives an identical digest,
  so two model versions trained on the same data are easy to spot.
- **Recoverable data.** Bucket object versioning keeps every overwritten parquet and
  model file, so the files behind an earlier model version can be restored by date.
- **Code versions.** Images are tagged with the git commit, which is also recorded on
  each MLflow run as `mlflow.source.git.commit`.
- **Serving lineage.** `GET /model/metadata` on the API returns an `mlflow` block with
  the run ID, model version and dataset digest of the model it is serving.

Known limitations, deliberately accepted for now:

- MLflow tracks the dataset's digest, not the data itself; recovering old data relies on
  bucket object versioning rather than a dataset-versioning tool such as DVC or lakeFS.
- MLflow uses its file-based store inside the bucket. MLflow 3 treats that store as
  maintenance-only and refuses it unless `MLFLOW_ALLOW_FILE_STORE=true` (set in the
  Dockerfile). The production-grade setup is an MLflow tracking server backed by
  Postgres (Cloud SQL) with `gs://` artifact storage.
- Cloud Storage FUSE has no file locking, which is safe here only because the pipeline
  job is the single writer.

## Prerequisites

- A Google Cloud project with billing enabled, and its project ID.
- The Google Cloud CLI: `winget install Google.CloudSDK`, then in a new terminal
  `gcloud auth login`.
- `configs/config.yaml` with your NOAA token (copy `configs/config.example.yaml`).
- Optional but recommended: a local `data/` from an earlier pipeline run. The setup
  script uploads it so the API works immediately and the first cloud run can skip
  ingestion.

## First deployment

Run from the repository root in PowerShell.

```powershell
# 1. One-off setup: APIs, image repository, bucket, service accounts, config secret,
#    and an initial upload of data/ (not mlruns/)
.\scripts\gcp_setup.ps1 -ProjectId <project-id>

# 2. Build the image with Cloud Build and deploy the API, job, MLflow UI and schedule
.\scripts\deploy.ps1 -ProjectId <project-id>

# 3. Train once on the uploaded data so MLflow has a run and model version 1.
#    Waits for the job, then restarts the API so it loads the new model.
.\scripts\run_pipeline.ps1 -ProjectId <project-id> -SkipIngestion -ForceTrain
```

Then check it:

```powershell
$api = gcloud run services describe agrisignal-api --region europe-west2 --format 'value(status.url)'
Invoke-RestMethod "$api/health"
Invoke-RestMethod "$api/model/metadata" | Select-Object -ExpandProperty mlflow
Invoke-RestMethod "$api/predict" -Method Post -ContentType 'application/json' -Body '{}'
```

Open the MLflow UI through an authenticated local proxy, then browse to
<http://localhost:5000>:

```powershell
gcloud run services proxy agrisignal-mlflow --region europe-west2 --port 5000
```

The experiment is `corn-futures-xgboost`; the registered model is under **Models**.

## Day to day

| Task | Command |
|---|---|
| Deploy code changes | `.\scripts\deploy.ps1 -ProjectId <id>` (rolls out the new image, keeps configuration) |
| Apply changes to the service/job configuration in `deploy.ps1` | `.\scripts\deploy.ps1 -ProjectId <id> -Recreate` |
| Run the full pipeline now (fresh downloads) | `.\scripts\run_pipeline.ps1 -ProjectId <id>` |
| Retrain now without downloading | `.\scripts\run_pipeline.ps1 -ProjectId <id> -SkipIngestion -ForceTrain` |
| Update `configs/config.yaml` in the cloud | re-run `.\scripts\gcp_setup.ps1 -ProjectId <id>` (adds a secret version) |
| List pipeline runs | `gcloud run jobs executions list --job agrisignal-pipeline --region europe-west2` |

The API loads the model and gold data when an instance starts. `run_pipeline.ps1`
restarts it after a run; after a *scheduled* run, a warm instance keeps serving the
previous model until it scales down or is redeployed.

## Running locally with MLflow 3

With a local `./mlruns` tracking URI, set the same opt-in the image uses, or training
still completes but skips MLflow logging with a warning:

```powershell
$env:MLFLOW_ALLOW_FILE_STORE = 'true'
python -m agrisignal.orchestration.flows.daily_pipeline --skip-ingestion --force-train
```

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `/health` shows `model_loaded: false`, `/predict` returns 503 | No model in the bucket yet. Run `run_pipeline.ps1 -SkipIngestion -ForceTrain`. |
| Cloud Build fails with a permissions error | The build service account lacks a role. Re-run `gcp_setup.ps1`, which grants Artifact Registry, logging and storage access. |
| Pipeline fails during ingestion | Yahoo Finance can rate-limit requests from cloud IP ranges. Retrain on existing data with `-SkipIngestion`. |
| `Permission denied` writing under `/app/data` | The bucket must be mounted with `uid=1000;gid=1000` (the image's non-root user); `deploy.ps1 -Recreate` reapplies it. |
| MLflow UI returns 403 | Host-header check; see `MLFLOW_SERVER_ALLOWED_HOSTS` in `docker/mlflow-ui.env.yaml`. |

## Tearing down

Deleting the project removes everything. To keep the project:

```powershell
gcloud scheduler jobs delete agrisignal-daily --location europe-west2 --quiet
gcloud run jobs delete agrisignal-pipeline --region europe-west2 --quiet
gcloud run services delete agrisignal-api --region europe-west2 --quiet
gcloud run services delete agrisignal-mlflow --region europe-west2 --quiet
gcloud storage rm --recursive gs://<project-id>-agrisignal-data
gcloud artifacts repositories delete agrisignal --location europe-west2 --quiet
gcloud secrets delete agrisignal-config --quiet
```
