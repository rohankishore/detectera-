# Use a specific Python version for reproducibility
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for Python libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first to leverage Docker cache
COPY detectra/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code into the container
# The .dockerignore file will exclude unnecessary files
COPY . .

# Create a non-root user for security
RUN useradd --create-home appuser && chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Set the working directory to where your app is
WORKDIR /app/detectra

# Make the entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Set environment variables
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Expose the port the app runs on
EXPOSE 5000

# Define the entrypoint for the container
ENTRYPOINT ["/app/entrypoint.sh"]