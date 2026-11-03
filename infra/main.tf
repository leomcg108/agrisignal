locals {
  # scripts/gcp_common.ps1 uses the same names; keep the two in step.
  data_bucket   = "${var.project_id}-agrisignal-data"
  config_secret = "agrisignal-config"

  # Cloud Run mounts volumes as root by default; uid/gid 1000 is the image's
  # non-root 'agrisignal' user, which has to write to the bucket.
  data_mount_options = ["uid=1000", "gid=1000"]

  apis = toset([
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "cloudscheduler.googleapis.com",
    "compute.googleapis.com",
    "iam.googleapis.com",
    "run.googleapis.com",
    "secretmanager.googleapis.com",
    "storage.googleapis.com",
  ])

  build_roles = toset([
    "roles/artifactregistry.writer",
    "roles/logging.logWriter",
    "roles/storage.objectViewer",
  ])
}

data "google_project" "this" {}

resource "google_project_service" "apis" {
  for_each = local.apis
  service  = each.key

  # Leave the API on if it is ever removed from this list or destroyed
  disable_on_destroy = false
}

# New projects run Cloud Build as the Compute Engine default service account.
# It usually has Editor already; these grants cover projects where it does not.
# On a brand-new project that account appears a minute or so after the Compute
# API is enabled, so the first apply can fail here; re-running it fixes that.
resource "google_project_iam_member" "build" {
  for_each = local.build_roles
  project  = var.project_id
  role     = each.key
  member   = "serviceAccount:${data.google_project.this.number}-compute@developer.gserviceaccount.com"

  depends_on = [google_project_service.apis]
}
