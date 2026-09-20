#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt"

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN=python
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
else
  echo "Error: neither 'python' nor 'python3' is available in PATH."
  exit 1
fi

echo "Installing Python requirements from ${REQUIREMENTS_FILE} using ${PYTHON_BIN}"
"${PYTHON_BIN}" -m pip install --break-system-packages -r "${REQUIREMENTS_FILE}"