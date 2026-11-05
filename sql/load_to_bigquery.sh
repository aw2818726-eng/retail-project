#!/usr/bin/env bash
# Load the four CSV files into BigQuery (project my-first-pro-504117, dataset egypt_retail).
# Requires the Google Cloud SDK and `gcloud auth login`. Run from the repository root.
set -euo pipefail

PROJECT="my-first-pro-504117"
DATASET="egypt_retail"
LOCATION="US"   # change to match your other datasets if needed

bq --project_id="$PROJECT" --location="$LOCATION" mk --dataset "$PROJECT:$DATASET" \
  || echo "Dataset already exists, continuing."

for table in customers products orders order_items; do
  bq --project_id="$PROJECT" --location="$LOCATION" load \
    --replace --source_format=CSV --skip_leading_rows=1 --autodetect \
    "$DATASET.$table" "data/$table.csv"
done

echo "Loaded. Now run sql/01_clean_views.sql, then the other queries in order."
