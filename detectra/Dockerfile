FROM python:3.11-slim

# Use /app as workdir
WORKDIR /app

# system deps for some libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libatlas-base-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# copy requirements and install
# Note: CI providers (like Render) often set the build context to the repo root while
# the Dockerfile may live in a subfolder; reference files with the repo-relative path
# so COPY works regardless of build context.
COPY detectra/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# copy app
# Copy only the `detectra` application into the image; model files are excluded from
# the repo for size reasons and are downloaded at runtime by `download_models.py`.
COPY detectra/ /app

# create a non-root user
RUN useradd --create-home appuser && chown -R appuser:appuser /app

WORKDIR /app

# ensure entrypoint executable
RUN chmod +x /app/entrypoint.sh || true

USER appuser

ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

EXPOSE 5000

ENTRYPOINT ["/app/entrypoint.sh"]
