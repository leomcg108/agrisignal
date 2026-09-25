<#
.SYNOPSIS
    Build the AgriSignal image and deploy it to Cloud Run: the public API,
    the pipeline job, the private MLflow UI, and the daily schedule.

.DESCRIPTION
    Run gcp_setup.ps1 once first. The first deploy creates each resource with
    its full configuration (bucket mount, config secret, env vars). Later
    deploys only roll out the new image and keep that configuration; pass
    -Recreate to delete and recreate the service and job so configuration
    changes in this script take effect. The API URL does not change.

.EXAMPLE
    .\scripts\deploy.ps1 -ProjectId my-project-id
#>
param(
    [Parameter(Mandatory = $true)] [string]$ProjectId,
    [string]$Region = 'europe-west2',
    [switch]$Recreate
)

$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\gcp_common.ps1"
Set-Location (Split-Path $PSScriptRoot -Parent)   # repo root: the build context

# Image tag = git commit; uncommitted changes get a -dirty-<timestamp> suffix
$GitSha = (git rev-parse --short HEAD | Out-String).Trim()
if (git status --porcelain) { $GitSha = "$GitSha-dirty-$(Get-Date -Format yyyyMMddHHmmss)" }
$Image = "${ImageBase}:$GitSha"

Invoke-Gcloud config set project $ProjectId

Write-Step "Building $Image with Cloud Build (about 5-10 minutes)"
Invoke-Gcloud builds submit --region $Region --config docker/cloudbuild.yaml `
    --substitutions "_IMAGE=$Image,_GIT_SHA=$GitSha" .

# The bucket is mounted where the code expects data/. Cloud Run mounts volumes
# as root by default; uid/gid 1000 is the image's non-root 'agrisignal' user.
$DataVolume  = "name=data,type=cloud-storage,bucket=$Bucket,mount-options=uid=1000;gid=1000"
$DataMount   = 'volume=data,mount-path=/app/data'
$ConfigMount = "/secrets/config.yaml=${ConfigSecret}:latest"

if ($Recreate) {
    Write-Step 'Deleting the services and job so they are recreated with full configuration'
    foreach ($svc in $ApiService, $MlflowService) {
        if (Test-Gcloud run services describe $svc --region $Region) {
            Invoke-Gcloud run services delete $svc --region $Region --quiet
        }
    }
    if (Test-Gcloud run jobs describe $PipelineJob --region $Region) {
        Invoke-Gcloud run jobs delete $PipelineJob --region $Region --quiet
    }
}

Write-Step "API service '$ApiService' (public)"
if (Test-Gcloud run services describe $ApiService --region $Region) {
    Invoke-Gcloud run deploy $ApiService --image $Image --region $Region --quiet
} else {
    Invoke-Gcloud run deploy $ApiService --image $Image --region $Region --quiet `
        --service-account $RunSa `
        --execution-environment gen2 `
        --cpu 1 --memory 2Gi --cpu-boost `
        --min-instances 0 --max-instances 2 `
        --allow-unauthenticated `
        --set-env-vars 'AGRISIGNAL_CONFIG=/secrets/config.yaml' `
        --set-secrets $ConfigMount `
        --add-volume $DataVolume `
        --add-volume-mount $DataMount
}

Write-Step "Pipeline job '$PipelineJob'"
if (Test-Gcloud run jobs describe $PipelineJob --region $Region) {
    Invoke-Gcloud run jobs update $PipelineJob --image $Image --region $Region --quiet
} else {
    # PREFECT_SERVER_ANALYTICS_ENABLED=false stops the flow's temporary
    # Prefect server logging telemetry errors against its own SQLite file.
    Invoke-Gcloud run jobs create $PipelineJob --image $Image --region $Region --quiet `
        --service-account $RunSa `
        --command 'python,-m,agrisignal.orchestration.flows.daily_pipeline' `
        --cpu 2 --memory 4Gi `
        --task-timeout 60m --max-retries 0 `
        --set-env-vars 'AGRISIGNAL_CONFIG=/secrets/config.yaml,MLFLOW_TRACKING_URI=file:///app/data/mlruns,PREFECT_SERVER_ANALYTICS_ENABLED=false' `
        --set-secrets $ConfigMount `
        --add-volume $DataVolume `
        --add-volume-mount $DataMount
}

Write-Step "MLflow UI service '$MlflowService' (private)"
if (Test-Gcloud run services describe $MlflowService --region $Region) {
    Invoke-Gcloud run deploy $MlflowService --image $Image --region $Region --quiet
} else {
    Invoke-Gcloud run deploy $MlflowService --image $Image --region $Region --quiet `
        --service-account $RunSa `
        --execution-environment gen2 `
        --command mlflow --args server --port 8080 `
        --cpu 1 --memory 2Gi --cpu-boost `
        --min-instances 0 --max-instances 1 `
        --no-allow-unauthenticated `
        --env-vars-file docker/mlflow-ui.env.yaml `
        --add-volume $DataVolume `
        --add-volume-mount $DataMount
}

Write-Step "Daily schedule '$SchedulerJob' (07:00 UTC, Mon-Fri, as orchestration.schedule_cron)"
Invoke-Gcloud run jobs add-iam-policy-binding $PipelineJob --region $Region `
    --member "serviceAccount:$SchedulerSa" --role roles/run.invoker --format none
$RunUri = "https://run.googleapis.com/v2/projects/$ProjectId/locations/$Region/jobs/${PipelineJob}:run"
$verb = 'create'
if (Test-Gcloud scheduler jobs describe $SchedulerJob --location $Region) { $verb = 'update' }
Invoke-Gcloud scheduler jobs $verb http $SchedulerJob --location $Region `
    --schedule '0 7 * * 1-5' --time-zone 'Etc/UTC' `
    --uri $RunUri --http-method POST `
    --oauth-service-account-email $SchedulerSa

$ApiUrl = Get-GcloudValue run services describe $ApiService --region $Region --format 'value(status.url)'

Write-Step 'Deployed'
Write-Host "Image:      $Image"
Write-Host "API:        $ApiUrl/health   (docs at $ApiUrl/docs)"
Write-Host "MLflow UI:  gcloud run services proxy $MlflowService --region $Region --port 5000"
Write-Host '            then open http://localhost:5000'
Write-Host ''
Write-Host 'First deploy? Train once on the uploaded data so MLflow has a run and model version:'
Write-Host "  .\scripts\run_pipeline.ps1 -ProjectId $ProjectId -SkipIngestion -ForceTrain"
