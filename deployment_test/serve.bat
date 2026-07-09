@echo off
setlocal enabledelayedexpansion

:: =====================================================================
:: EdgeGPT Serve
::
:: Starts the llama.cpp HTTP server with an EdgeGPT GGUF model and
:: opens the Web UI (http://localhost:8080).
::
:: Usage:  serve.bat  [MODEL.gguf]  [PORT]
::
:: Close this window (or press Ctrl+C) to stop the server cleanly.
:: =====================================================================

set "LLAMA_SERVER=D:\workspace\llamacpp\source\llama.cpp\build-cuda\bin\llama-server.exe"

:: Defaults -- override by passing arguments or editing below.
set "MODEL=%~dp0edgegpt-tinystories-f16.gguf"
set "PORT=8080"
set "HOST=0.0.0.0"
set "CTX=2048"
set "TEMP=0.8"

:: ---- Parse optional arguments ---------------------------------------
if not "%~1"=="" set "MODEL=%~1"
if not "%~2"=="" set "PORT=%~2"

:: ---- Guard: model file must exist -----------------------------------
if not exist "%MODEL%" (
    echo [ERROR] Model file not found: %MODEL%
    echo.
    echo Available models:
    for %%f in ("%~dp0*.gguf") do echo   %%~nxf
    pause
    exit /b 1
)

:: ---- Guard: llama-server must exist ---------------------------------
if not exist "%LLAMA_SERVER%" (
    echo [ERROR] llama-server.exe not found: %LLAMA_SERVER%
    echo Build it: cmake -B build-cuda -DGGML_CUDA=ON ^&^& cmake --build build-cuda --config Release
    pause
    exit /b 1
)

:: ---- Guard: port already in use? ------------------------------------
curl -s http://localhost:%PORT%/health >nul 2>&1
if not errorlevel 1 (
    echo [WARNING] Something is already listening on port %PORT%.
    echo           If it is another llama-server, close that window first.
    echo.
    choice /c yn /m "Start anyway? [y/n]"
    if errorlevel 2 exit /b 1
)

:: ---- Launch server (background, same console) -----------------------
echo.
echo ==========================================================
echo               EdgeGPT LLM Server
echo ----------------------------------------------------------
echo   Model:  %MODEL%
echo   URL:    http://localhost:%PORT%
echo   Close this window to stop the server
echo ==========================================================
echo.

start "" /b "%LLAMA_SERVER%" ^
    -m "%MODEL%" ^
    --host %HOST% ^
    --port %PORT% ^
    -c %CTX% ^
    --temp %TEMP%

:: ---- Wait for the server to be ready --------------------------------
echo Waiting for server to be ready...
set "READY=0"
for /l %%i in (1,1,30) do (
    curl -s http://localhost:%PORT%/health >nul 2>&1
    if !errorlevel! equ 0 (
        set "READY=1"
        goto :ready
    )
    ping -n 2 127.0.0.1 >nul
)
:ready

if "%READY%"=="1" (
    echo Server is ready.
) else (
    echo [WARNING] Server did not respond within 30 s -- opening browser anyway.
)

:: ---- Open the Web UI ------------------------------------------------
start "" http://localhost:%PORT%

echo.
echo ==========================================================
echo   PRESS ANY KEY or CLOSE THIS WINDOW to stop
echo ==========================================================
echo.

:: ---- Wait for user --------------------------------------------------
pause >nul

:: ---- Cleanup --------------------------------------------------------
echo.
echo Stopping server...
taskkill /f /im llama-server.exe >nul 2>&1
echo Server stopped.
endlocal
