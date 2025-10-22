# Deployment guide

This document explains simple ways to deploy the Detectra Flask app.

1) Docker (recommended, portable)

Build the image from repository root (where `Dockerfile` is):

    docker build -t detectra:latest .

Run it locally exposing port 5000:

    docker run --rm -p 5000:5000 detectra:latest

Notes:
- The Docker image excludes large pickles by default (.dockerignore). If you keep model pickles out of the repo, add a step to fetch them at container startup (e.g. download from S3).

2) Heroku (quick, free tier available)

- Create a `Procfile` with: `web: gunicorn wsgi:application --bind 0.0.0.0:$PORT --workers 3`
- Add `requirements.txt` and `runtime.txt` (e.g. `python-3.11.6`).
- `git push heroku master`

3) Google Cloud Run / Azure Container Instances / AWS ECS

- Push your Docker image to a container registry (Docker Hub, GCR, ACR, ECR), then deploy the container.

4) Azure Web Apps (Linux)

- Use the container deployment or deploy from a GitHub Actions workflow that builds and pushes the image.

Model storage and secrets
- Don't store large pickles in git; use an object store (S3, GCS, Azure Blob) or Git LFS.
- Put secrets (e.g., SECRET_KEY, S3 credentials) in environment variables — your deployment target will provide a secret manager.

Static files & persistent state
- The container's filesystem is ephemeral. If the app writes reports or uploaded files, either mount a volume or upload them to object storage.

CI/CD
- Consider adding a GitHub Actions workflow that builds and publishes the Docker image and optionally deploys it.
