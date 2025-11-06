#!/bin/bash

set -e

echo "--- Starting Grape Analysis Pipeline ---"

if [ -d "venv" ]; then
	# Activate the virtual environment (Linux syntax)
	# shellcheck disable=SC1091
	source venv/bin/activate
fi

echo "Running analysis once..."
python main.py watch --run-once

echo "--- Demo Finished ---"