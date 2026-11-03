# Deploying AgriSignal to Google Cloud Run

AgriSignal runs on Google Cloud as one container image deployed three ways, with a
Cloud Storage bucket standing in for the local `data/` directory. Terraform (`infra/`)
provisions and configures every resource; the PowerShell scripts build the image, run
Terraform, and handle the steps Terraform does not own.

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
| `<project>-tfstate` | GCS bucket holding Terraform state (versioned). Created by `gcp_setup.ps1`, not by Terraform. |

Region: `europe-west2` (London). The docker-compose extras (Postgres, Prefect server,
Prometheus, Grafana) are not deployed: Cloud Scheduler and Cloud Run jobs replace the
Prefect server for this single daily flow, which runs Prefect with a temporary local
server inside the job.

## Infrastructure as code

`infra/` is a single Terraform configuration with state in `gs://<project>-tfstate`.

| File | What it manages |
|---|---|
| `main.tf` | Enabled APIs, and Cloud Build's grants on the Compute Engine default service account |
| `storage.tf` | Artifact Registry repository, data bucket (versioned, `prevent_destroy`), config secret |
| `iam.tf` | The two service accounts and every IAM grant |
| `run.tf` | The API and MLflow services and the pipeline job: image, env, CPU/memory, mounts |
| `scheduler.tf` | The daily Cloud Scheduler trigger |

Deliberately outside Terraform:

- **Image builds.** `deploy.ps1` builds with Cloud Build and passes the tag to
  Terraform as `-var image=...`.
- **The config secret's contents.** Terraform creates the secret, but `gcp_setup.ps1`
  uploads `configs/config.yaml` with gcloud, so the NOAA token never lands in
  Terraform state.
- **Data.** The first upload of `data/`, and running the pipeline.
- **The state bucket**, which has to exist before Terraform can run.

IAM is granted one member at a time (`*_iam_member`), never with the authoritative
`_binding` or `_policy` resources, so Terraform cannot strip grants it does not know
about. gcloud writes two things Terraform ignores: the `client` fields on Cloud Run
resources, and the API's `model-refreshed-at` label that `run_pipeline.ps1` sets to
restart it.

To see what Terraform would change without deploying (the scripts run `init` for you;
by hand it is needed once per clone):

```powershell
cd infra
terraform init -backend-config="bucket=<project-id>-tfstate"
terraform plan -var project_id=<project-id> -var image=$(terraform output -raw image)
```

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
- Terraform 1.7 or later: `winget install Hashicorp.Terraform`, then
  `gcloud auth application-default login` (Terraform authenticates with these
  credentials, not with `gcloud auth login`).
- `configs/config.yaml` with your NOAA token (copy `configs/config.example.yaml`).
- Optional but recommended: a local `data/` from an earlier pipeline run. The setup
  script uploads it so the API works immediately and the first cloud run can skip
  ingestion.

## First deployment

Run from the repository root in PowerShell. Each Terraform apply shows its plan and
waits for `yes`; pass `-AutoApprove` to skip that.

```powershell
# 1. One-off setup: the Terraform state bucket; then Terraform creates the image
#    repository, data bucket and config secret; then config.yaml is uploaded to the
#    secret and data/ (not mlruns/) to the bucket
.\scripts\gcp_setup.ps1 -ProjectId <project-id>

# 2. Build the image with Cloud Build, then terraform apply creates everything else:
#    service accounts, the API, job, MLflow UI and schedule
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
| Deploy code changes | `.\scripts\deploy.ps1 -ProjectId <id>` |
| Change infrastructure (env vars, CPU/memory, schedule, IAM, ...) | Edit `infra/*.tf`, then `.\scripts\deploy.ps1 -ProjectId <id>` |
| Preview infrastructure changes | `terraform plan` (see [Infrastructure as code](#infrastructure-as-code)) |
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
| First `gcp_setup.ps1` on a new project fails on `google_project_iam_member.build` | The Compute Engine default service account appears a minute or so after its API is enabled. Re-run the script. |
| Terraform: `Error acquiring the state lock` | Another apply is running, or one was interrupted. If none is running, run `terraform force-unlock <lock-id>` in `infra/`. |
| Pipeline fails during ingestion | Yahoo Finance can rate-limit requests from cloud IP ranges. Retrain on existing data with `-SkipIngestion`. |
| `Permission denied` writing under `/app/data` | The bucket must be mounted with `uid=1000`, `gid=1000` (the image's non-root user); see `data_mount_options` in `infra/main.tf`. |
| MLflow UI returns 403 | Host-header check; see `MLFLOW_SERVER_ALLOWED_HOSTS` in `infra/run.tf`. |

## Tearing down

Deleting the project removes everything, including the Terraform state.

To keep the project: `terraform destroy` is blocked on purpose, because the data bucket
has `prevent_destroy` and the Cloud Run services and job have deletion protection on.
First remove the `lifecycle { prevent_destroy = true }` block in `infra/storage.tf` and
add `deletion_protection = false` to the three resources in `infra/run.tf`. Then:

```powershell
cd infra
$vars = @('-var=project_id=<project-id>', "-var=image=$(terraform output -raw image)")
terraform apply @vars                                            # records deletion_protection = false
gcloud storage rm --recursive gs://<project-id>-agrisignal-data   # versioned data, then the bucket
terraform destroy @vars                                          # leaves the APIs enabled
gcloud storage rm --recursive gs://<project-id>-tfstate
```
