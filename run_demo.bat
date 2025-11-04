@echo OFF
echo --- Starting Grape Analysis Pipeline ---

REM Activate the virtual environment
call venv\Scripts\activate

REM Run the main Python script
echo Running analysis...
python main.py --input data/sample_image.hdr

echo --- Demo Finished ---
pause