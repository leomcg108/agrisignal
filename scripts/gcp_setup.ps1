<#
.SYNOPSIS
    One-off Google Cloud setup for AgriSignal: APIs, image repository, data
    bucket, service accounts, config secret, and an initial copy of data/.

.DESCRIPTION
    Safe to re-run: resources that already exist are left alone. Re-running
    uploads configs/config.yaml as a new secret version, so use it after
    changing the config too. Run deploy.ps1 afterwards.

.EXAMPLE
    .\scripts\gcp_setup.ps1 -ProjectId my-project-id
#>
param(
    [Parameter(Mandatory = $true)] [string]$ProjectId,
    [string]$Region = 'europe-west2'
)

$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\gcp_common.ps1"
Set-Location (Split-Path $PSScriptRoot -Parent)   # repo root, so relative paths resolve

if (-not (Test-Path configs/config.yaml)) {
    throw 'configs/config.yaml not found. Copy configs/config.example.yaml and add your NOAA token first.'
}

Write-Step "Project $ProjectId"
Invoke-Gcloud config set project $ProjectId
$ProjectNumber = Get-GcloudValue projects describe $ProjectId --format 'value(projectNumber)'

Write-Step 'Enabling APIs (a minute or two the first time)'
Invoke-Gcloud services enable `
    run.googleapis.com `
    artifactregistry.googleapis.com `
    cloudbuild.googleapis.com `
    secretmanager.googleapis.com `
    cloudscheduler.googleapis.com `
    storage.googleapis.com `
    iam.googleapis.com `
    compute.googleapis.com

Write-Step "Artifact Registry repository '$RepoName'"
if (-not (Test-Gcloud artifacts repositories describe $RepoName --location $Region)) {
    Invoke-Gcloud artifacts repositories create $RepoName `
        --repository-format docker --location $Region --description 'AgriSignal images'
}

Write-Step "Data bucket gs://$Bucket"
if (-not (Test-Gcloud storage buckets describe "gs://$Bucket")) {
    Invoke-Gcloud storage buckets create "gs://$Bucket" `
        --location $Region --uniform-bucket-level-access --public-access-prevention
}
# Every overwrite of a dataset or model file keeps the previous version recoverable
Invoke-Gcloud storage buckets update "gs://$Bucket" --versioning

Write-Step 'Service accounts'
$accounts = @(
    @{ Name = 'agrisignal-run';       Display = 'AgriSignal API, pipeline job and MLflow UI' },
    @{ Name = 'agrisignal-scheduler'; Display = 'Triggers the daily AgriSignal pipeline job' }
)
foreach ($sa in $accounts) {
    if (-not (Test-Gcloud iam service-accounts describe "$($sa.Name)@$ProjectId.iam.gserviceaccount.com")) {
        Invoke-Gcloud iam service-accounts create $sa.Name --display-name $sa.Display
    }
}

Write-Step 'Bucket read/write for the runtime service account'
Invoke-Gcloud storage buckets add-iam-policy-binding "gs://$Bucket" `
    --member "serviceAccount:$RunSa" --role roles/storage.objectUser --format none

Write-Step "Config secret '$ConfigSecret' from configs/config.yaml"
if (Test-Gcloud secrets describe $ConfigSecret) {
    Invoke-Gcloud secrets versions add $ConfigSecret --data-file configs/config.yaml
} else {
    Invoke-Gcloud secrets create $ConfigSecret --data-file configs/config.yaml --replication-policy automatic
}
Invoke-Gcloud secrets add-iam-policy-binding $ConfigSecret `
    --member "serviceAccount:$RunSa" --role roles/secretmanager.secretAccessor --format none

Write-Step 'Cloud Build permissions'
# New projects run builds as the Compute Engine default service account. It
# usually has Editor already; these grants cover projects where it does not.
$BuildSa = "$ProjectNumber-compute@developer.gserviceaccount.com"
$found = $false
foreach ($attempt in 1..6) {
    if (Test-Gcloud iam service-accounts describe $BuildSa) { $found = $true; break }
    Write-Host "Waiting for $BuildSa to be created..."
    Start-Sleep -Seconds 10
}
if ($found) {
    foreach ($role in 'roles/artifactregistry.writer', 'roles/logging.logWriter', 'roles/storage.objectViewer') {
        Invoke-Gcloud projects add-iam-policy-binding $ProjectId `
            --member "serviceAccount:$BuildSa" --role $role --condition None --format none
    }
} else {
    Write-Warning "$BuildSa not found; skipping build grants. If the build fails on permissions, re-run this script."
}

Write-Step 'Seeding the bucket with local data/ (first run only)'
if (Test-Gcloud storage objects describe "gs://$Bucket/gold/gold_features.parquet") {
    Write-Host 'Bucket already has gold data; not uploading.'
} elseif (Test-Path data/gold/gold_features.parquet) {
    # Local mlruns/ is deliberately not uploaded: its runs record C:\ paths,
    # so the cloud MLflow history starts fresh.
    foreach ($layer in 'bronze', 'silver', 'gold', 'models') {
        if (Test-Path "data/$layer") {
            Invoke-Gcloud storage cp --recursive "data/$layer" "gs://$Bucket/"
        }
    }
} else {
    Write-Warning 'No local data/ to upload. Run the pipeline job with ingestion to fill the bucket.'
}

Write-Step 'Setup complete'
Write-Host "Next: .\scripts\deploy.ps1 -ProjectId $ProjectId"
