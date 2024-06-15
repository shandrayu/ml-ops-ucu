#!/bin/bash

if ! command -v jq &> /dev/null; then
    echo "jq could not be found. Please install jq to proceed."
    exit 1
fi

CREDENTIALS_FILE="credentials.json"
if [ ! -f "$CREDENTIALS_FILE" ]; then
    echo "Credentials file $CREDENTIALS_FILE not found!"
    exit 1
fi

MINIO_ACCESS_KEY=$(jq -r '.accessKey' $CREDENTIALS_FILE)
MINIO_SECRET_KEY=$(jq -r '.secretKey' $CREDENTIALS_FILE)

CURRENT_DIR=$(pwd)

# Create the .env file
{
    echo "PROJECT_DIR=$CURRENT_DIR"
    echo "MINIO_ACCESS_KEY=$MINIO_ACCESS_KEY"
    echo "MINIO_SECRET_KEY=$MINIO_SECRET_KEY"
} > .env

echo ".env file has been created with PROJECT_DIR set to $CURRENT_DIR"
