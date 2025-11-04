#!/bin/bash

echo "--- Starting Grape Analysis Pipeline ---"

# Activate the virtual environment (Linux syntax)
source venv/bin/activate

# Run the main Python script
echo "Running analysis..."
python main.py #<-- Make sure this is your main script and correct path

echo "--- Demo Finished ---"