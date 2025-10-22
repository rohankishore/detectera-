FROM python:3.11-slim

WORKDIR /app

# system deps for some libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libatlas-base-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# copy app
# Copy application code but exclude model files (they can be downloaded at runtime)
COPY . /app

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
