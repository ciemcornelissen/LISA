@echo OFF
echo --- Starting Grape Analysis Pipeline ---

REM Activate the virtual environment if present
IF EXIST venv\Scripts\activate.bat CALL venv\Scripts\activate.bat

REM Run the pipeline once using the new CLI
echo Running analysis...
python main.py watch --run-once

echo --- Demo Finished ---
pause