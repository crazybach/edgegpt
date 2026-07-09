<#
.SYNOPSIS
    Serve an EdgeGPT GGUF model via llama.cpp with the built-in Web UI.

.DESCRIPTION
    Starts llama-server.exe, opens the Web UI in your default browser,
    and shuts down the server cleanly when you press Ctrl+C or close
    the PowerShell window.

.PARAMETER Model
    Path to the .gguf model file.  Defaults to the tinystories-f16 model
    in the same directory as this script.

.PARAMETER Port
    Port to listen on (default: 8080).

.PARAMETER HostAddr
    IP address to bind (default: 0.0.0.0, accessible from LAN).

.PARAMETER CtxSize
    Context size in tokens (default: 2048).

.PARAMETER Temp
    Sampling temperature (default: 0.8).

.EXAMPLE
    .\serve.ps1

.EXAMPLE
    .\serve.ps1 -Model .\edgegpt-tinystories-f32.gguf -Port 9090

.EXAMPLE
    .\serve.ps1 -Model .\edgegpt-Q5_K_M.gguf -Temp 1.0
#>

param(
    [string]$Model    = "$PSScriptRoot\edgegpt-tinystories-f16.gguf",
    [int]   $Port     = 8080,
    [string]$HostAddr = "0.0.0.0",
    [int]   $CtxSize  = 2048,
    [float] $Temp     = 0.8
)

# ---- paths ------------------------------------------------------------
$LlamaServer = "D:\workspace\llamacpp\source\llama.cpp\build-cuda\bin\llama-server.exe"
$ServerUrl   = "http://localhost:$Port"

# =======================================================================
# Guards
# =======================================================================

if (-not (Test-Path $Model)) {
    Write-Host "[ERROR] Model file not found: $Model" -ForegroundColor Red
    Write-Host ""
    Write-Host "Available models:"
    Get-ChildItem "$PSScriptRoot\*.gguf" | ForEach-Object { Write-Host "  $($_.Name)" }
    exit 1
}

if (-not (Test-Path $LlamaServer)) {
    Write-Host "[ERROR] llama-server.exe not found: $LlamaServer" -ForegroundColor Red
    Write-Host "Build it: cmake -B build-cuda -DGGML_CUDA=ON && cmake --build build-cuda --config Release"
    exit 1
}

# Check for existing llama-server process.
$existing = Get-Process -Name "llama-server" -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "[WARNING] llama-server is already running (PID $($existing.Id))." -ForegroundColor Yellow
    $answer = Read-Host "Stop it and restart? [y/N]"
    if ($answer -match '^[yY]') {
        Stop-Process -Name "llama-server" -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 1
    }
    else {
        Write-Host "Leaving existing server alone. Opening browser..."
        Start-Process $ServerUrl
        exit 0
    }
}

# =======================================================================
# Start the server
# =======================================================================

Write-Host ""
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "              EdgeGPT LLM Server" -ForegroundColor Cyan
Write-Host "----------------------------------------------------------" -ForegroundColor Cyan
Write-Host "  Model:  $Model" -ForegroundColor Cyan
Write-Host "  URL:    $ServerUrl" -ForegroundColor Cyan
Write-Host "  Ctrl+C or close window to stop" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ""

$proc = Start-Process -FilePath $LlamaServer `
    -ArgumentList @(
        "-m", $Model,
        "--host", $HostAddr,
        "--port", $Port,
        "-c", $CtxSize,
        "--temp", $Temp
    ) `
    -PassThru -NoNewWindow

# =======================================================================
# Cleanup handler -- runs on Ctrl+C, window close, or normal exit
# =======================================================================

function Stop-Server {
    if ($proc -and (-not $proc.HasExited)) {
        Write-Host "`nStopping EdgeGPT server (PID $($proc.Id))..." -ForegroundColor Yellow
        # Send graceful close first, then force-kill after timeout.
        $proc.CloseMainWindow() | Out-Null
        $exited = $proc.WaitForExit(3000)
        if (-not $exited) {
            Write-Host "Server did not exit gracefully - force-killing..." -ForegroundColor Red
            $proc.Kill()
            $proc.WaitForExit(5000) | Out-Null
        }
        Write-Host "Server stopped." -ForegroundColor Green
    }
}

# PowerShell.Exiting fires on normal exit AND window close (CTRL_CLOSE_EVENT).
Register-EngineEvent -SourceIdentifier PowerShell.Exiting -Action {
    # Must re-derive process -- the script variable is not captured.
    $p = Get-Process -Name "llama-server" -ErrorAction SilentlyContinue
    if ($p) {
        $p.CloseMainWindow() | Out-Null
        $done = $p.WaitForExit(3000)
        if (-not $done) { $p.Kill() }
    }
} | Out-Null

# =======================================================================
# Wait for server readiness + open browser
# =======================================================================

try {
    Write-Host "Waiting for server to be ready..." -ForegroundColor Cyan
    $ready = $false
    for ($i = 0; $i -lt 30; $i++) {
        try {
            $null = Invoke-WebRequest -Uri "$ServerUrl/health" -UseBasicParsing -TimeoutSec 2
            $ready = $true
            break
        }
        catch {
            Start-Sleep -Seconds 1
        }
    }

    if ($ready) {
        Write-Host "Server is ready at $ServerUrl" -ForegroundColor Green
    }
    else {
        Write-Host "[WARNING] Server did not respond within 30 s." -ForegroundColor Yellow
    }

    # Open the Web UI.
    Start-Process $ServerUrl

    Write-Host ""
    Write-Host "=========================================================="
    Write-Host "  PRESS Ctrl+C or CLOSE THIS WINDOW to stop"
    Write-Host "=========================================================="
    Write-Host ""

    # Block until the server process exits (user killed it or it crashed).
    $proc.WaitForExit()
    Write-Host "Server process exited (code $($proc.ExitCode))." -ForegroundColor Yellow

}
catch {
    # Ctrl+C lands here.
    Write-Host "`nInterrupted." -ForegroundColor Yellow
}
finally {
    Stop-Server
}
