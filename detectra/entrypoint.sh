#!/bin/sh
set -e

# If MODEL_BASE_URL and MODEL_FILES are provided, download the models into MODEL_DEST_DIR
if [ -n "$MODEL_BASE_URL" ] && [ -n "$MODEL_FILES" ]; then
  echo "Fetching models from $MODEL_BASE_URL"
  if ! python download_models.py; then
    echo "Model download script failed"
    if [ "$FAIL_IF_MISSING" = "1" ] || [ "$FAIL_IF_MISSING" = "true" ]; then
      echo "FAIL_IF_MISSING set — exiting container"
      exit 2
    else
      echo "Continuing without all models (set FAIL_IF_MISSING=1 to fail startup)"
    fi
  fi
else
  echo "MODEL_BASE_URL or MODEL_FILES not set; skipping model download"
fi

echo "Starting Gunicorn..."
# Use PORT from environment (Render provides $PORT). Default to 5000 when not set.
PORT_TO_BIND="${PORT:-5000}"
echo "Binding to 0.0.0.0:${PORT_TO_BIND}"
exec gunicorn wsgi:application --bind 0.0.0.0:${PORT_TO_BIND} --workers 3
