@echo off
setlocal

title EdgeGPT llama.cpp server - http://127.0.0.1:8081/

set "LLAMA_SERVER=D:\workspace\llamacpp\source\llama.cpp\build-cuda\bin\llama-server.exe"
set "MODEL=%~dp0artifacts\exports\improvement_1_full_1b-f16.gguf"
set "CHAT_TEMPLATE=%~dp0configs\llama_base_completion.jinja"
set "URL=http://127.0.0.1:8081/"

if not exist "%LLAMA_SERVER%" (
    echo ERROR: llama-server.exe was not found:
    echo   %LLAMA_SERVER%
    pause
    exit /b 1
)

if not exist "%MODEL%" (
    echo ERROR: GGUF model was not found:
    echo   %MODEL%
    pause
    exit /b 1
)

if not exist "%CHAT_TEMPLATE%" (
    echo ERROR: Base completion template was not found:
    echo   %CHAT_TEMPLATE%
    pause
    exit /b 1
)

echo Starting EdgeGPT with CUDA llama.cpp...
echo Web UI: %URL%
echo.
echo Press Ctrl+C or close this window to stop the server.
echo.

start "" /b powershell.exe -NoProfile -WindowStyle Hidden -Command ^
    "Start-Sleep -Seconds 3; Start-Process '%URL%'"

"%LLAMA_SERVER%" ^
    --model "%MODEL%" ^
    --host 127.0.0.1 ^
    --port 8081 ^
    --ctx-size 2048 ^
    --gpu-layers 99 ^
    --parallel 1 ^
    --jinja ^
    --chat-template-file "%CHAT_TEMPLATE%"

echo.
echo llama.cpp server stopped.
endlocal
