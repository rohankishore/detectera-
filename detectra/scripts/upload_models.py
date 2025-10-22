#!/usr/bin/env python3
"""Upload model files to S3 using boto3.

Usage:
  python scripts/upload_models.py --bucket my-bucket --prefix models file1.pkl file2.pkl

This will upload each file to s3://my-bucket/<prefix>/<filename>
"""
import argparse
import os
import boto3


def upload_file(s3_client, local_path, bucket, key):
    print(f"Uploading {local_path} -> s3://{bucket}/{key}")
    s3_client.upload_file(local_path, bucket, key)
    print("Done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', required=True)
    parser.add_argument('--prefix', default='models')
    parser.add_argument('files', nargs='+', help='Local paths to upload')
    args = parser.parse_args()

    s3 = boto3.client('s3')

    for f in args.files:
        if not os.path.exists(f):
            print(f"File not found: {f}")
            continue
        fname = os.path.basename(f)
        key = f"{args.prefix.rstrip('/')}/{fname}"
        upload_file(s3, f, args.bucket, key)


if __name__ == '__main__':
    main()
