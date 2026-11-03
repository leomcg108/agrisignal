terraform {
  required_version = ">= 1.7"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 8.9"
    }
  }

  # State lives in gs://<project-id>-tfstate, created by scripts/gcp_setup.ps1.
  # The bucket is passed at init time because backend blocks cannot use variables:
  #   terraform init -backend-config="bucket=<project-id>-tfstate"
  backend "gcs" {
    prefix = "agrisignal"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region

  # Bill API quota to this project rather than the quota project of whoever runs
  # Terraform (gcloud's application-default login picks one up from gcloud config).
  billing_project       = var.project_id
  user_project_override = true
}
