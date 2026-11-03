output "api_url" {
  description = "Public URL of the API."
  value       = google_cloud_run_v2_service.api.uri
}

output "image" {
  description = "Image currently deployed; pass it back with -var to plan without a rebuild."
  value       = google_cloud_run_v2_service.api.template[0].containers[0].image
}

output "image_repository" {
  description = "Image name without a tag; scripts/deploy.ps1 tags it with the git commit."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}/agrisignal"
}

output "data_bucket" {
  description = "Bucket mounted at /app/data."
  value       = google_storage_bucket.data.name
}

output "mlflow_proxy_command" {
  description = "Opens the private MLflow UI at http://localhost:5000."
  value       = "gcloud run services proxy ${google_cloud_run_v2_service.mlflow.name} --region ${var.region} --project ${var.project_id} --port 5000"
}
