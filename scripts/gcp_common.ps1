# ================================================================
# Shared resource names and helpers for the GCP scripts.
# Dot-sourced by gcp_setup.ps1, deploy.ps1 and run_pipeline.ps1 after
# they define $ProjectId and $Region; not meant to be run directly.
# ================================================================

$Bucket        = "$ProjectId-agrisignal-data"     # replaces the local data/ directory
$RepoName      = 'agrisignal'                      # Artifact Registry repository
$ImageBase     = "$Region-docker.pkg.dev/$ProjectId/$RepoName/agrisignal"
$ConfigSecret  = 'agrisignal-config'               # configs/config.yaml (holds the NOAA token)
$RunSa         = "agrisignal-run@$ProjectId.iam.gserviceaccount.com"
$SchedulerSa   = "agrisignal-scheduler@$ProjectId.iam.gserviceaccount.com"
$ApiService    = 'agrisignal-api'
$MlflowService = 'agrisignal-mlflow'
$PipelineJob   = 'agrisignal-pipeline'
$SchedulerJob  = 'agrisignal-daily'

if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    throw 'gcloud not found. Install it with: winget install Google.CloudSDK (then open a new terminal and run: gcloud auth login)'
}

function Write-Step([string]$Message) {
    Write-Host ''
    Write-Host "==> $Message" -ForegroundColor Cyan
}

# PowerShell does not stop on a failing native command, so every gcloud call
# goes through one of these. ErrorActionPreference is relaxed inside them so
# that gcloud's progress output on stderr is not mistaken for an error.

function Invoke-Gcloud {
    # Run gcloud; stop the script if it fails.
    $ErrorActionPreference = 'Continue'
    & gcloud @args
    if ($LASTEXITCODE -ne 0) { throw "gcloud $($args -join ' ') failed (exit code $LASTEXITCODE)" }
}

function Get-GcloudValue {
    # Run gcloud and return its trimmed stdout; stop the script if it fails.
    $ErrorActionPreference = 'Continue'
    $out = & gcloud @args
    if ($LASTEXITCODE -ne 0) { throw "gcloud $($args -join ' ') failed (exit code $LASTEXITCODE)" }
    return ($out | Out-String).Trim()
}

function Test-Gcloud {
    # True if the gcloud command succeeds. Used for "does this exist?" checks.
    $ErrorActionPreference = 'Continue'
    & gcloud @args *> $null
    return ($LASTEXITCODE -eq 0)
}
