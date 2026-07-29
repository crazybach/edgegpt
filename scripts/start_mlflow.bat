@echo off
setlocal
cd /d "%~dp0.."

set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

set "BACKEND_URI=sqlite:///artifacts/mlflow.db"
set "ARTIFACT_ROOT=%CD%\artifacts\mlflow-artifacts"

if not exist "%ARTIFACT_ROOT%" mkdir "%ARTIFACT_ROOT%"

echo MLflow dashboard: http://127.0.0.1:5001
echo Close this window or press Ctrl+C to stop the server.
start "" "http://127.0.0.1:5001"

"%PYTHON%" -m mlflow server ^
  --backend-store-uri "%BACKEND_URI%" ^
  --default-artifact-root "%ARTIFACT_ROOT%" ^
  --host 127.0.0.1 ^
  --port 5001 ^
  --workers 1

endlocal
