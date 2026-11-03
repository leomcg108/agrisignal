# One image, deployed three ways. All three mount the data bucket at /app/data,
# where the code expects data/, which needs the gen2 execution environment.

# Public FastAPI service. It loads the model and gold data when an instance
# starts; scripts/run_pipeline.ps1 forces a new revision after a run by setting
# the model-refreshed-at label, which Terraform leaves alone.
resource "google_cloud_run_v2_service" "api" {
  name     = "agrisignal-api"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account       = google_service_account.run.email
    execution_environment = "EXECUTION_ENVIRONMENT_GEN2"

    scaling {
      min_instance_count = 0
      max_instance_count = 2
    }

    containers {
      image = var.image

      env {
        name  = "AGRISIGNAL_CONFIG"
        value = "/secrets/config.yaml"
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "2Gi"
        }
        cpu_idle          = true # CPU is billed only while handling requests
        startup_cpu_boost = true
      }

      volume_mounts {
        name       = "config"
        mount_path = "/secrets"
      }
      volume_mounts {
        name       = "data"
        mount_path = "/app/data"
      }
    }

    volumes {
      name = "config"
      secret {
        secret = google_secret_manager_secret.config.secret_id
        items {
          version = "latest"
          path    = "config.yaml"
        }
      }
    }
    volumes {
      name = "data"
      gcs {
        bucket        = google_storage_bucket.data.name
        read_only     = false
        mount_options = local.data_mount_options
      }
    }
  }

  lifecycle {
    # client* record which tool last deployed (gcloud for run_pipeline.ps1)
    ignore_changes = [client, client_version, template[0].labels["model-refreshed-at"]]
  }

  depends_on = [
    google_secret_manager_secret_iam_member.run_config,
    google_storage_bucket_iam_member.run_data,
  ]
}

# Daily pipeline: ingest -> silver -> gold -> quality -> train (Mondays).
resource "google_cloud_run_v2_job" "pipeline" {
  name     = "agrisignal-pipeline"
  location = var.region

  template {
    template {
      service_account       = google_service_account.run.email
      execution_environment = "EXECUTION_ENVIRONMENT_GEN2"
      timeout               = "3600s"
      max_retries           = 0

      containers {
        image   = var.image
        command = ["python", "-m", "agrisignal.orchestration.flows.daily_pipeline"]

        env {
          name  = "AGRISIGNAL_CONFIG"
          value = "/secrets/config.yaml"
        }
        env {
          name  = "MLFLOW_TRACKING_URI"
          value = "file:///app/data/mlruns"
        }
        # Stops the flow's temporary Prefect server logging telemetry errors
        # against its own SQLite file
        env {
          name  = "PREFECT_SERVER_ANALYTICS_ENABLED"
          value = "false"
        }

        resources {
          limits = {
            cpu    = "2"
            memory = "4Gi"
          }
        }

        volume_mounts {
          name       = "config"
          mount_path = "/secrets"
        }
        volume_mounts {
          name       = "data"
          mount_path = "/app/data"
        }
      }

      volumes {
        name = "config"
        secret {
          secret = google_secret_manager_secret.config.secret_id
          items {
            version = "latest"
            path    = "config.yaml"
          }
        }
      }
      volumes {
        name = "data"
        gcs {
          bucket        = google_storage_bucket.data.name
          read_only     = false
          mount_options = local.data_mount_options
        }
      }
    }
  }

  lifecycle {
    ignore_changes = [client, client_version]
  }

  depends_on = [
    google_secret_manager_secret_iam_member.run_config,
    google_storage_bucket_iam_member.run_data,
  ]
}

# MLflow UI over the same file store the pipeline writes. Private: there is no
# allUsers invoker grant, so open it with `gcloud run services proxy`.
resource "google_cloud_run_v2_service" "mlflow" {
  name     = "agrisignal-mlflow"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account       = google_service_account.run.email
    execution_environment = "EXECUTION_ENVIRONMENT_GEN2"

    scaling {
      min_instance_count = 0
      max_instance_count = 1
    }

    containers {
      image   = var.image
      command = ["mlflow"]
      args    = ["server"]

      # `mlflow server` reads all of its options from these variables
      env {
        name  = "MLFLOW_BACKEND_STORE_URI"
        value = "file:///app/data/mlruns"
      }
      env {
        name  = "MLFLOW_HOST"
        value = "0.0.0.0"
      }
      env {
        name  = "MLFLOW_PORT"
        value = "8080"
      }
      # Host-header allowlist: the service's run.app URL, and localhost for
      # `gcloud run services proxy`
      env {
        name  = "MLFLOW_SERVER_ALLOWED_HOSTS"
        value = "*.run.app,localhost:*,127.0.0.1:*"
      }
      env {
        name  = "MLFLOW_WORKERS"
        value = "1"
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "2Gi"
        }
        cpu_idle          = true # CPU is billed only while handling requests
        startup_cpu_boost = true
      }

      volume_mounts {
        name       = "data"
        mount_path = "/app/data"
      }
    }

    volumes {
      name = "data"
      gcs {
        bucket        = google_storage_bucket.data.name
        read_only     = false
        mount_options = local.data_mount_options
      }
    }
  }

  lifecycle {
    ignore_changes = [client, client_version]
  }

  depends_on = [google_storage_bucket_iam_member.run_data]
}
