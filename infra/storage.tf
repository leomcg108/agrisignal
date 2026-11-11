resource "google_artifact_registry_repository" "images" {
  repository_id = "agrisignal"
  location      = var.region
  format        = "DOCKER"
  description   = "AgriSignal images"

  # Each deploy pushes a ~0.9 GB image. Keep the 5 newest and delete the rest
  # (a KEEP policy overrides DELETE), so only those 5 can be rolled back to.
  # The deployed image is always among them: only deploy.ps1 pushes, and it
  # deploys what it pushes. Cleanup runs in the background about once a day.
  cleanup_policy_dry_run = false

  cleanup_policies {
    id     = "keep-5-newest"
    action = "KEEP"
    most_recent_versions {
      keep_count = 5
    }
  }

  cleanup_policies {
    id     = "delete-older"
    action = "DELETE"
    condition {
      tag_state = "ANY"
    }
  }

  depends_on = [google_project_service.apis]
}

# Replaces the local data/ directory: bronze/ silver/ gold/ models/ mlruns/
resource "google_storage_bucket" "data" {
  name                        = local.data_bucket
  location                    = upper(var.region)
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  # Every overwrite of a dataset or model file keeps the previous version recoverable
  versioning {
    enabled = true
  }

  # Holds every dataset and the MLflow history; never delete it from Terraform
  lifecycle {
    prevent_destroy = true
  }

  depends_on = [google_project_service.apis]
}

# Holds configs/config.yaml, which includes the NOAA token. Terraform manages the
# secret but not its contents, so the token never enters Terraform state;
# scripts/gcp_setup.ps1 uploads each new version.
resource "google_secret_manager_secret" "config" {
  secret_id = local.config_secret

  replication {
    auto {}
  }

  depends_on = [google_project_service.apis]
}
