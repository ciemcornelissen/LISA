#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="data"
ARCHIVE_PATH="${TARGET_DIR}/sample_capture.tar.gz"
DEFAULT_URL="https://cloud.ilabt.imec.be/index.php/s/5z7pC8Dwt7LfHWL/download"

if [[ -z "${LISA_SAMPLE_URL:-}" ]]; then
  echo "LISA_SAMPLE_URL not set, using default ${DEFAULT_URL}" >&2
  LISA_SAMPLE_URL="${DEFAULT_URL}"
fi

mkdir -p "${TARGET_DIR}"

if [[ -f "${ARCHIVE_PATH}" ]]; then
  echo "Removing existing archive at ${ARCHIVE_PATH}" >&2
  rm -f "${ARCHIVE_PATH}"
fi

echo "Downloading sample data from ${LISA_SAMPLE_URL}"
curl -L "${LISA_SAMPLE_URL}" -o "${ARCHIVE_PATH}"

echo "Extracting sample capture into ${TARGET_DIR}"
tar -xzf "${ARCHIVE_PATH}" -C "${TARGET_DIR}"

echo "Cleaning up archive"
rm -f "${ARCHIVE_PATH}"

echo "Sample capture available under ${TARGET_DIR}"
