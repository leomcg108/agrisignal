<#
.SYNOPSIS
    Build the AgriSignal image and deploy it with Terraform: the public API,
    the pipeline job, the private MLflow UI, and the daily schedule.

.DESCRIPTION
    Run gcp_setup.ps1 once first. Every deploy builds a new image with Cloud
    Build, then runs `terraform apply` on infra/ with that image, so changes to
    the configuration in infra/ roll out the same way as code changes.
    Terraform shows the plan and waits for "yes" unless -AutoApprove is given.
    The API URL does not change.

.EXAMPLE
    .\scripts\deploy.ps1 -ProjectId my-project-id
#>
param(
    [Parameter(Mandatory = $true)] [string]$ProjectId,
    [string]$Region = 'europe-west2',
    [switch]$AutoApprove
)

$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\gcp_common.ps1"
Set-Location (Split-Path $PSScriptRoot -Parent)   # repo root: the build context

# Image tag = git commit; uncommitted changes get a -dirty-<timestamp> suffix
$GitSha = (git rev-parse --short HEAD | Out-String).Trim()
if (git status --porcelain) { $GitSha = "$GitSha-dirty-$(Get-Date -Format yyyyMMddHHmmss)" }
$Image = "${ImageBase}:$GitSha"

Write-Step 'Terraform init'
Initialize-Terraform

Write-Step "Building $Image with Cloud Build (about 5-10 minutes)"
Invoke-Gcloud builds submit --project $ProjectId --region $Region --config docker/cloudbuild.yaml `
    --substitutions "_IMAGE=$Image,_GIT_SHA=$GitSha" .

Write-Step "Terraform apply with $GitSha"
$apply = @('apply', "-var=project_id=$ProjectId", "-var=region=$Region", "-var=image=$Image")
if ($AutoApprove) { $apply += '-auto-approve' }
Invoke-Terraform @apply

$ApiUrl = Get-TerraformOutput api_url

Write-Step 'Deployed'
Write-Host "Image:      $Image"
Write-Host "API:        $ApiUrl/health   (docs at $ApiUrl/docs)"
Write-Host "MLflow UI:  $(Get-TerraformOutput mlflow_proxy_command)"
Write-Host '            then open http://localhost:5000'
Write-Host ''
Write-Host 'First deploy? Train once on the uploaded data so MLflow has a run and model version:'
Write-Host "  .\scripts\run_pipeline.ps1 -ProjectId $ProjectId -SkipIngestion -ForceTrain"
