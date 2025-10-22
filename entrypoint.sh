#!/bin/sh
set -e

# If MODEL_BASE_URL and MODEL_FILES are provided, download the models into MODEL_DEST_DIR
if [ -n "$MODEL_BASE_URL" ] && [ -n "$MODEL_FILES" ]; then
  echo "Fetching models from $MODEL_BASE_URL"
  python download_models.py
else
  echo "MODEL_BASE_URL or MODEL_FILES not set; skipping model download"
fi

echo "Starting Gunicorn..."
exec gunicorn wsgi:application --bind 0.0.0.0:5000 --workers 3
