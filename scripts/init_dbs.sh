#!/bin/bash
# Creates each database listed in POSTGRES_MULTIPLE_DATABASES (comma-separated).
# The postgres image runs this once, on first start with an empty data volume.
set -euo pipefail

if [ -n "${POSTGRES_MULTIPLE_DATABASES:-}" ]; then
  for db in $(echo "$POSTGRES_MULTIPLE_DATABASES" | tr ',' ' '); do
    echo "Creating database: $db"
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres \
      -c "CREATE DATABASE \"$db\" OWNER \"$POSTGRES_USER\";"
  done
fi
