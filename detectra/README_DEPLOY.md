Quick deployment checklist

This file contains minimal, copy-paste commands and env var suggestions to deploy the `detectra` app (the Flask app in this folder) to Render or to build locally with Docker.

1) Local Docker build (from repo root) — validate image builds

```bash
# from repo root (where detectra/ lives)
# Tag can be anything; here we use detectra:local
docker build -f detectra/Dockerfile -t detectra:local .

docker run --rm -p 5000:5000 \
# Run the container and forward port 5000
docker run --rm -p 5000:5000 \
  -e MODEL_BASE_URL="" \
  -e MODEL_FILES="" \
  -e PORT=5000 \
  detectra:local
```

2) Upload model files to S3 (recommended) — example using AWS CLI

```bash
# Upload pickled models to s3://my-bucket/detectra-models/
aws s3 cp detectra/drug_binary_xgb.pkl s3://my-bucket/detectra-models/drug_binary_xgb.pkl
aws s3 cp detectra/drug_multiclass_xgb.pkl s3://my-bucket/detectra-models/drug_multiclass_xgb.pkl
aws s3 cp detectra/drug_label_encoder.pkl s3://my-bucket/detectra-models/drug_label_encoder.pkl
aws s3 cp detectra/model/ensemble_classifier_chains.pkl s3://my-bucket/detectra-models/ensemble_classifier_chains.pkl
```

3) Render.com deploy

- Create a new Web Service on Render and connect your GitHub repo `eldho-1/render_dectectra`.
- Build command: leave default (Dockerfile is used)
- Start command: leave empty (entrypoint handles start)
- Set environment variables in the Render dashboard (Environment -> Environment Variables):
  - MODEL_BASE_URL: s3://my-bucket/detectra-models
  - MODEL_FILES: drug_binary_xgb.pkl,drug_multiclass_xgb.pkl,drug_label_encoder.pkl,ensemble_classifier_chains.pkl
  - MODEL_DEST_DIR: model
  - S3_ACCESS_KEY_ID and S3_SECRET_ACCESS_KEY (if using a private bucket) OR configure AWS IAM role
  - SECRET_KEY: a secure random string for Flask sessions
- Deploy; check the service logs for the `Downloading` messages from `download_models.py`.

Note about PORT on Render:
- Render provides a dynamic port via the `PORT` environment variable. `entrypoint.sh` was updated to bind Gunicorn to `$PORT` (fallback: 5000) so the container will work on Render without modification.

Notes:
- If you prefer public hosting for models, set MODEL_BASE_URL to the HTTP(S) base URL.
- Consider using Git LFS for model pickles if you want them in repo history (not recommended).
