<#
.SYNOPSIS
    Run the AgriSignal pipeline job on Cloud Run now, wait for it, then
    restart the API so it serves the model and data the run just wrote.

.DESCRIPTION
    The API loads the model and gold features once at startup, so after a
    run it is moved to a new revision to pick them up. The scheduled daily
    run does not do this; its instances refresh when they next cold-start.

.EXAMPLE
    # First run after deploying: train on the data uploaded by gcp_setup.ps1
    .\scripts\run_pipeline.ps1 -ProjectId my-project-id -SkipIngestion -ForceTrain

.EXAMPLE
    # Normal run: download fresh NOAA and futures data (trains on Mondays)
    .\scripts\run_pipeline.ps1 -ProjectId my-project-id
#>
param(
    [Parameter(Mandatory = $true)] [string]$ProjectId,
    [string]$Region = 'europe-west2',
    [switch]$SkipIngestion,
    [switch]$ForceTrain
)

$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\gcp_common.ps1"

$pipelineArgs = @()
if ($SkipIngestion) { $pipelineArgs += '--skip-ingestion' }
if ($ForceTrain)    { $pipelineArgs += '--force-train' }

$execute = @('run', 'jobs', 'execute', $PipelineJob, '--region', $Region, '--project', $ProjectId, '--wait')
if ($pipelineArgs.Count -gt 0) { $execute += "--args=$($pipelineArgs -join ',')" }

Write-Step "Running $PipelineJob $($pipelineArgs -join ' ') (logs: Cloud Console > Cloud Run > Jobs)"
Invoke-Gcloud @execute

Write-Step "Restarting $ApiService on a new revision to load the new model"
Invoke-Gcloud run services update $ApiService --region $Region --project $ProjectId --quiet `
    --update-env-vars "MODEL_REFRESHED_AT=$(Get-Date -Format yyyyMMddHHmmss)"

$ApiUrl = Get-GcloudValue run services describe $ApiService --region $Region --project $ProjectId --format 'value(status.url)'
Write-Host ''
Write-Host "Check what the API serves: $ApiUrl/model/metadata  (see the 'mlflow' block)"
