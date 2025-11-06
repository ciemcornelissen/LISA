#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="models"
DEFAULT_URL="https://cloud.ilabt.imec.be/index.php/s/TpySxxyaTpQqLzz/download"
ARCHIVE_BASENAME="${LISA_MODELS_ARCHIVE_NAME:-LISA_models.zip}"
ARCHIVE_PATH="${TARGET_DIR}/${ARCHIVE_BASENAME}"

if [[ -z "${LISA_MODELS_URL:-}" ]]; then
  echo "LISA_MODELS_URL not set, using default ${DEFAULT_URL}" >&2
  LISA_MODELS_URL="${DEFAULT_URL}"
fi

echo "Fetching pretrained models from ${LISA_MODELS_URL}" >&2
mkdir -p "${TARGET_DIR}"

if [[ -f "${ARCHIVE_PATH}" ]]; then
  echo "Removing existing archive at ${ARCHIVE_PATH}" >&2
  rm -f "${ARCHIVE_PATH}"
fi

echo "Clearing previous model artefacts under ${TARGET_DIR}" >&2
find "${TARGET_DIR}" -mindepth 1 -maxdepth 1 -exec rm -rf {} +

curl -L "${LISA_MODELS_URL}" -o "${ARCHIVE_PATH}"

case "${ARCHIVE_PATH}" in
  *.zip)
    echo "Extracting zip archive into ${TARGET_DIR}" >&2
    unzip -oq "${ARCHIVE_PATH}" -d "${TARGET_DIR}"
    ;;
  *.tar.gz|*.tgz)
    echo "Extracting tar archive into ${TARGET_DIR}" >&2
    tar -xzf "${ARCHIVE_PATH}" -C "${TARGET_DIR}"
    ;;
  *)
    echo "Unsupported archive format: ${ARCHIVE_PATH}" >&2
    exit 1
    ;;
esac

rm -f "${ARCHIVE_PATH}"

# Flatten "models/" if the archive expanded into a single top-level directory.
if [[ $(find "${TARGET_DIR}" -mindepth 1 -maxdepth 1 -type f | wc -l) -eq 0 ]]; then
  _subdirs=()
  while IFS= read -r -d '' dir; do
    _subdirs+=("${dir}")
  done < <(find "${TARGET_DIR}" -mindepth 1 -maxdepth 1 -type d -print0)
  if [[ ${#_subdirs[@]} -eq 1 ]]; then
    echo "Flattening extracted directory ${_subdirs[0]}" >&2
    shopt -s dotglob nullglob
    mv "${_subdirs[0]}/"* "${TARGET_DIR}/"
    shopt -u dotglob nullglob
    rmdir "${_subdirs[0]}"
  fi
  unset _subdirs
fi

echo "Models ready under ${TARGET_DIR}" >&2
