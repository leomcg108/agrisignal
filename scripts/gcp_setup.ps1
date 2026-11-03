<#
.SYNOPSIS
    One-off Google Cloud setup for AgriSignal: the Terraform state bucket, the
    resources needed before the first image build, the config secret's
    contents, and an initial copy of data/.

.DESCRIPTION
    Terraform (infra/) owns the cloud resources. This script creates what
    Terraform itself needs (a state bucket and two APIs), then applies only the
    image repository, data bucket, config secret and Cloud Build grants, since
    the Cloud Run services cannot be created until deploy.ps1 has built an
    image. deploy.ps1 applies everything else.

    Safe to re-run: Terraform leaves unchanged resources alone. Re-running
    uploads configs/config.yaml as a new secret version, so use it after
    changing the config too.

.EXAMPLE
    .\scripts\gcp_setup.ps1 -ProjectId my-project-id
#>
param(
    [Parameter(Mandatory = $true)] [string]$ProjectId,
    [string]$Region = 'europe-west2',
    [switch]$AutoApprove
)

$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\gcp_common.ps1"
Set-Location (Split-Path $PSScriptRoot -Parent)   # repo root, so relative paths resolve

if (-not (Test-Path configs/config.yaml)) {
    throw 'configs/config.yaml not found. Copy configs/config.example.yaml and add your NOAA token first.'
}

Write-Step "APIs Terraform needs before it can run (project $ProjectId)"
Invoke-Gcloud services enable --project $ProjectId `
    cloudresourcemanager.googleapis.com `
    serviceusage.googleapis.com `
    storage.googleapis.com

Write-Step "Terraform state bucket gs://$StateBucket"
if (-not (Test-Gcloud storage buckets describe "gs://$StateBucket" --project $ProjectId)) {
    Invoke-Gcloud storage buckets create "gs://$StateBucket" --project $ProjectId `
        --location $Region --uniform-bucket-level-access --public-access-prevention
}
# Every state change keeps the previous state recoverable
Invoke-Gcloud storage buckets update "gs://$StateBucket" --project $ProjectId --versioning

Write-Step 'Terraform init'
Initialize-Terraform

Write-Step 'Terraform: image repository, data bucket, config secret, Cloud Build grants'
# -target is deliberate: the services and job need an image, which needs this
# repository first. Their image is not known yet, so pass whatever is deployed
# (a placeholder on a new project; the targeted resources do not use it).
$Deployed = "${ImageBase}:not-yet-built"
if (Test-Gcloud run services describe $ApiService --region $Region --project $ProjectId) {
    $Deployed = Get-GcloudValue run services describe $ApiService --region $Region --project $ProjectId `
        --format 'value(spec.template.spec.containers[0].image)'
}
$apply = @('apply', "-var=project_id=$ProjectId", "-var=region=$Region", "-var=image=$Deployed",
    '-target=google_artifact_registry_repository.images',
    '-target=google_storage_bucket.data',
    '-target=google_secret_manager_secret.config',
    '-target=google_project_iam_member.build')
if ($AutoApprove) { $apply += '-auto-approve' }
Invoke-Terraform @apply

Write-Step "Config secret '$ConfigSecret': uploading configs/config.yaml as a new version"
# Uploaded here rather than by Terraform so the NOAA token stays out of Terraform state
Invoke-Gcloud secrets versions add $ConfigSecret --project $ProjectId --data-file configs/config.yaml

Write-Step 'Seeding the bucket with local data/ (first run only)'
if (Test-Gcloud storage objects describe "gs://$Bucket/gold/gold_features.parquet" --project $ProjectId) {
    Write-Host 'Bucket already has gold data; not uploading.'
} elseif (Test-Path data/gold/gold_features.parquet) {
    # Local mlruns/ is deliberately not uploaded: its runs record C:\ paths,
    # so the cloud MLflow history starts fresh.
    foreach ($layer in 'bronze', 'silver', 'gold', 'models') {
        if (Test-Path "data/$layer") {
            Invoke-Gcloud storage cp --project $ProjectId --recursive "data/$layer" "gs://$Bucket/"
        }
    }
} else {
    Write-Warning 'No local data/ to upload. Run the pipeline job with ingestion to fill the bucket.'
}

Write-Step 'Setup complete'
Write-Host "Next: .\scripts\deploy.ps1 -ProjectId $ProjectId"
