variable "project_id" {
  description = "Google Cloud project ID."
  type        = string
}

variable "region" {
  description = "Region for every regional resource."
  type        = string
  default     = "europe-west2"
}

variable "image" {
  description = "Full image reference to run, e.g. <region>-docker.pkg.dev/<project>/agrisignal/agrisignal:<git-sha>. Set by scripts/deploy.ps1."
  type        = string
}
