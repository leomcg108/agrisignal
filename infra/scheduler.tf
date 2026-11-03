# 07:00 UTC, Mon-Fri, matching orchestration.schedule_cron in the config
resource "google_cloud_scheduler_job" "daily" {
  name      = "agrisignal-daily"
  region    = var.region
  schedule  = "0 7 * * 1-5"
  time_zone = "Etc/UTC"

  http_target {
    uri         = "https://run.googleapis.com/v2/${google_cloud_run_v2_job.pipeline.id}:run"
    http_method = "POST"

    oauth_token {
      service_account_email = google_service_account.scheduler.email
    }
  }

  # Cloud Scheduler does not retry a failed trigger; neither does the job (max_retries = 0)
  retry_config {
    retry_count = 0
  }

  depends_on = [google_cloud_run_v2_job_iam_member.scheduler_invoker]
}
