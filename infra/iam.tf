resource "google_service_account" "run" {
  account_id   = "agrisignal-run"
  display_name = "AgriSignal API, pipeline job and MLflow UI"

  depends_on = [google_project_service.apis]
}

resource "google_service_account" "scheduler" {
  account_id   = "agrisignal-scheduler"
  display_name = "Triggers the daily AgriSignal pipeline job"

  depends_on = [google_project_service.apis]
}

# Single-member grants (_iam_member) only: the _binding and _policy resources
# replace everything else on the role or resource.

resource "google_storage_bucket_iam_member" "run_data" {
  bucket = google_storage_bucket.data.name
  role   = "roles/storage.objectUser"
  member = google_service_account.run.member
}

resource "google_secret_manager_secret_iam_member" "run_config" {
  secret_id = google_secret_manager_secret.config.id
  role      = "roles/secretmanager.secretAccessor"
  member    = google_service_account.run.member
}

resource "google_cloud_run_v2_service_iam_member" "api_public" {
  name     = google_cloud_run_v2_service.api.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}

resource "google_cloud_run_v2_job_iam_member" "scheduler_invoker" {
  name     = google_cloud_run_v2_job.pipeline.name
  location = var.region
  role     = "roles/run.invoker"
  member   = google_service_account.scheduler.member
}
