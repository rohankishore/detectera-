import os
import sys
import urllib.request

MODEL_BASE_URL = os.environ.get("MODEL_BASE_URL")
MODEL_FILES = os.environ.get("MODEL_FILES")  # comma-separated filenames
DEST_DIR = os.environ.get("MODEL_DEST_DIR", "model")

def download_file(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        print(f"Downloading {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)
        print(f"Saved {dest}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        raise

def main():
    if not MODEL_BASE_URL or not MODEL_FILES:
        print("MODEL_BASE_URL or MODEL_FILES not set; skipping model download.")
        return

    files = [f.strip() for f in MODEL_FILES.split(',') if f.strip()]
    if not files:
        print("No model files listed in MODEL_FILES; skipping.")
        return

    for fname in files:
        url = MODEL_BASE_URL.rstrip('/') + '/' + fname
        dest = os.path.join(DEST_DIR, fname)
        download_file(url, dest)

if __name__ == '__main__':
    main()
