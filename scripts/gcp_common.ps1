# ================================================================
# Shared resource names and helpers for the GCP scripts.
# Dot-sourced by gcp_setup.ps1, deploy.ps1 and run_pipeline.ps1 after
# they define $ProjectId and $Region; not meant to be run directly.
#
# Terraform (infra/) creates and configures the resources. These names
# mirror infra/; they are repeated here so the scripts work before
# Terraform has any state and without running Terraform at all.
# ================================================================

$Bucket        = "$ProjectId-agrisignal-data"     # replaces the local data/ directory
$StateBucket   = "$ProjectId-tfstate"             # Terraform state, created by gcp_setup.ps1
$RepoName      = 'agrisignal'                      # Artifact Registry repository
$ImageBase     = "$Region-docker.pkg.dev/$ProjectId/$RepoName/agrisignal"
$ConfigSecret  = 'agrisignal-config'               # configs/config.yaml (holds the NOAA token)
$ApiService    = 'agrisignal-api'
$PipelineJob   = 'agrisignal-pipeline'
$InfraDir      = Join-Path (Split-Path $PSScriptRoot -Parent) 'infra'

if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    throw 'gcloud not found. Install it with: winget install Google.CloudSDK (then open a new terminal and run: gcloud auth login)'
}

function Write-Step([string]$Message) {
    Write-Host ''
    Write-Host "==> $Message" -ForegroundColor Cyan
}

# PowerShell does not stop on a failing native command, so every gcloud and
# terraform call goes through one of these. ErrorActionPreference is relaxed
# inside them so that progress output on stderr is not mistaken for an error.

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

function Invoke-Terraform {
    # Run terraform in infra/; stop the script if it fails.
    $ErrorActionPreference = 'Continue'
    & terraform "-chdir=$InfraDir" @args
    if ($LASTEXITCODE -ne 0) { throw "terraform $($args -join ' ') failed (exit code $LASTEXITCODE)" }
}

function Get-TerraformOutput([string]$Name) {
    $ErrorActionPreference = 'Continue'
    $out = & terraform "-chdir=$InfraDir" output -raw $Name
    if ($LASTEXITCODE -ne 0) { throw "terraform output $Name failed (exit code $LASTEXITCODE)" }
    return ($out | Out-String).Trim()
}

function Initialize-Terraform {
    # Point infra/ at this project's state bucket. -reconfigure makes switching
    # projects safe: state is remote, so nothing local needs migrating.
    if (-not (Get-Command terraform -ErrorAction SilentlyContinue)) {
        throw 'terraform not found. Install it with: winget install Hashicorp.Terraform (then open a new terminal and run: gcloud auth application-default login)'
    }
    Invoke-Terraform init -input=false -reconfigure "-backend-config=bucket=$StateBucket"
}
