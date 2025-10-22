import os
import sys
import urllib.request
from urllib.parse import urlparse

# Environment configuration
MODEL_BASE_URL = os.environ.get("MODEL_BASE_URL")
# MODEL_FILES can be comma-separated names OR full URLs. Examples:
#   drug_x.pkl,drug_y.pkl
#   https://example.com/path/drug_x.pkl,https://...
MODEL_FILES = os.environ.get("MODEL_FILES")  # comma-separated filenames or URLs
DEST_DIR = os.environ.get("MODEL_DEST_DIR", "model")
# If set to '1' or 'true', script will exit non-zero when any required model fails to download
FAIL_IF_MISSING = os.environ.get('FAIL_IF_MISSING', '0').lower() in ('1', 'true', 'yes')

def download_file_public(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        print(f"Downloading {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)
        print(f"Saved {dest}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        raise

def download_file_s3(bucket, key, dest, aws_access_key=None, aws_secret_key=None, region=None):
    # Use boto3 if available (for private buckets)
    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError
    except Exception as e:
        raise RuntimeError("boto3 required for S3 downloads but it's not installed")

    session_kwargs = {}
    if aws_access_key and aws_secret_key:
        session_kwargs['aws_access_key_id'] = aws_access_key
        session_kwargs['aws_secret_access_key'] = aws_secret_key
    if region:
        session_kwargs['region_name'] = region

    s = boto3.Session(**session_kwargs) if session_kwargs else boto3
    s3 = s.resource('s3') if hasattr(s, 'resource') else boto3.resource('s3')
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        print(f"Downloading s3://{bucket}/{key} -> {dest}")
        s3.Bucket(bucket).download_file(key, dest)
        print(f"Saved {dest}")
    except Exception as e:
        print(f"Failed to download s3://{bucket}/{key}: {e}")
        raise

def main():
    if not MODEL_BASE_URL or not MODEL_FILES:
        print("MODEL_BASE_URL or MODEL_FILES not set; skipping model download.")
        return

    raw_items = [f.strip() for f in MODEL_FILES.split(',') if f.strip()]
    files = []
    # Normalize: if an item is a URL (has scheme), keep as full URL; else treat as filename
    for item in raw_items:
        parsed = urlparse(item)
        if parsed.scheme and parsed.netloc:
            files.append({'url': item, 'name': os.path.basename(parsed.path)})
        else:
            files.append({'name': item, 'url': None})
    if not files:
        print("No model files listed in MODEL_FILES; skipping.")
        return

    # Check if MODEL_BASE_URL looks like an S3 URL
    is_s3 = False
    if MODEL_BASE_URL:
        is_s3 = MODEL_BASE_URL.startswith('s3://') or 's3.amazonaws.com' in MODEL_BASE_URL

    # Optional AWS credentials (for private buckets)
    aws_access_key = os.environ.get('S3_ACCESS_KEY_ID') or os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.environ.get('S3_SECRET_ACCESS_KEY') or os.environ.get('AWS_SECRET_ACCESS_KEY')
    aws_region = os.environ.get('AWS_REGION')

    failures = []
    for info in files:
        name = info.get('name')
        url = info.get('url')

        # If the item was a full URL, download directly
        if url:
            dest = os.path.join(DEST_DIR, name)
            try:
                download_file_public(url, dest)
            except Exception:
                failures.append(url)
            continue

        # Otherwise build URL/key from MODEL_BASE_URL
        try:
            if is_s3:
                # Normalize s3://bucket/path or https://.../bucket/path
                if MODEL_BASE_URL.startswith('s3://'):
                    base = MODEL_BASE_URL[len('s3://'):].rstrip('/')
                    bucket, *prefix = base.split('/', 1)
                    prefix = prefix[0] if prefix else ''
                    key = f"{prefix.rstrip('/')}/{name}" if prefix else name
                else:
                    # https://bucket.s3.amazonaws.com/path or https://s3.amazonaws.com/bucket/path
                    urlbase = MODEL_BASE_URL.rstrip('/')
                    parts = urlbase.split('s3.amazonaws.com/')
                    if len(parts) > 1:
                        base = parts[1]
                        bucket, *prefix = base.split('/', 1)
                        prefix = prefix[0] if prefix else ''
                        key = f"{prefix.rstrip('/')}/{name}" if prefix else name
                    else:
                        raise RuntimeError("Cannot parse S3 bucket from MODEL_BASE_URL")

                dest = os.path.join(DEST_DIR, name)
                download_file_s3(bucket, key, dest, aws_access_key, aws_secret_key, aws_region)
            else:
                url = MODEL_BASE_URL.rstrip('/') + '/' + name
                dest = os.path.join(DEST_DIR, name)
                download_file_public(url, dest)
        except Exception:
            failures.append(name)

    if failures:
        print(f"Failed to download the following items: {failures}")
        if FAIL_IF_MISSING:
            print("FAIL_IF_MISSING is set — exiting with error")
            sys.exit(2)
        else:
            print("Continuing despite missing files (set FAIL_IF_MISSING=1 to fail container startup)")

if __name__ == '__main__':
    main()
