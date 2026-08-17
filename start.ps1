[CmdletBinding()]
param(
    [switch]$Reload,
    [switch]$NoBrowser,
    [switch]$SkipInstall,
    [int]$Port = 8765,
    [int]$StartupTimeoutSeconds = 900
)

# Keep this launcher ASCII-only. Windows PowerShell 5.1 otherwise misreads a
# UTF-8 script without a BOM and can fail before showing the real error.
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$uv = Get-Command uv -ErrorAction SilentlyContinue
$baseUrl = "http://127.0.0.1:$Port"
$startupMutex = [System.Threading.Mutex]::new($false, "Local\RealTimeTranslation.MeetingV2.Startup")
$ownsStartupMutex = $false
$serverProcess = $null
$pollIntervalMilliseconds = 500
$maxWaitAttempts = [Math]::Ceiling(($StartupTimeoutSeconds * 1000) / $pollIntervalMilliseconds)

function Get-MeetingServiceState {
    param([string]$BaseUrl)

    try {
        $response = Invoke-RestMethod -Uri "$BaseUrl/api/v2/health" -TimeoutSec 2
        if ($null -eq $response.status -or $null -eq $response.capabilities -or $null -eq $response.languages) {
            return $null
        }
        return $response
    } catch {
        return $null
    }
}

function Test-PortListening {
    param([int]$TargetPort)

    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $result = $client.BeginConnect("127.0.0.1", $TargetPort, $null, $null)
        if (-not $result.AsyncWaitHandle.WaitOne(500)) { return $false }
        $client.EndConnect($result)
        return $true
    } catch {
        return $false
    } finally {
        $client.Close()
    }
}

try {
    try {
        $ownsStartupMutex = $startupMutex.WaitOne(0)
    } catch [System.Threading.AbandonedMutexException] {
        $ownsStartupMutex = $true
    }

    if (-not $ownsStartupMutex) {
        Write-Host "Another Meeting v2 launcher is starting the service; waiting for it..."
        for ($attempt = 0; $attempt -lt $maxWaitAttempts; $attempt++) {
            $existingService = Get-MeetingServiceState -BaseUrl $baseUrl
            if ($existingService -and $existingService.status -eq "ready") {
                Write-Host "Meeting v2 is already running and models are ready at $baseUrl"
                if (-not $NoBrowser) { Start-Process $baseUrl | Out-Null }
                return
            }
            if ($startupMutex.WaitOne($pollIntervalMilliseconds)) {
                $ownsStartupMutex = $true
                break
            }
            Start-Sleep -Milliseconds $pollIntervalMilliseconds
        }
        if (-not $ownsStartupMutex) {
            throw "Another Meeting v2 launcher did not become ready within $StartupTimeoutSeconds seconds."
        }
    }

    $existingService = Get-MeetingServiceState -BaseUrl $baseUrl
    if ($existingService) {
        if ($existingService.status -eq "ready") {
            Write-Host "Meeting v2 is already running and models are ready at $baseUrl"
            if (-not $NoBrowser) { Start-Process $baseUrl | Out-Null }
            return
        }

        Write-Host "Meeting v2 is already running; waiting for models to finish loading..."
        for ($attempt = 0; $attempt -lt $maxWaitAttempts; $attempt++) {
            Start-Sleep -Milliseconds $pollIntervalMilliseconds
            $existingService = Get-MeetingServiceState -BaseUrl $baseUrl
            if ($existingService -and $existingService.status -eq "ready") {
                Write-Host "Meeting v2 models are ready at $baseUrl"
                if (-not $NoBrowser) { Start-Process $baseUrl | Out-Null }
                return
            }
            if (-not $existingService -and -not (Test-PortListening -TargetPort $Port)) { break }
        }
        if ($existingService) {
            $message = if ($existingService.message) { $existingService.message } else { $existingService.status }
            throw "Meeting v2 did not become ready within $StartupTimeoutSeconds seconds: $message"
        }
    }

    if (Test-PortListening -TargetPort $Port) {
        throw "Port $Port is already in use by another application. Close it or launch Meeting v2 with a different -Port value."
    }

    if (-not $uv) {
        throw "uv was not found. Install uv or create the Python 3.11 environment manually."
    }

    if (-not (Test-Path -LiteralPath $venvPython)) {
        & $uv.Source venv --python 3.11 .venv
        if ($LASTEXITCODE -ne 0) { throw "Creating the Python 3.11 virtual environment failed." }
    }

    if (-not $SkipInstall) {
        Write-Host "Installing v2 dependencies..."
        & $uv.Source pip install --python $venvPython -e ".[audio,dev]"
        if ($LASTEXITCODE -ne 0) { throw "Installing v2 dependencies failed." }
    }

    $envPath = Join-Path $projectRoot ".env"
    $envExamplePath = Join-Path $projectRoot ".env.example"
    if (-not (Test-Path -LiteralPath $envPath)) {
        if (-not (Test-Path -LiteralPath $envExamplePath)) {
            throw "Neither .env nor .env.example exists."
        }
        Copy-Item -LiteralPath $envExamplePath -Destination $envPath
        Write-Host "Created .env from .env.example."
    }

    $logDir = Join-Path $projectRoot "result"
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    $stdoutLog = Join-Path $logDir "server.stdout.log"
    $stderrLog = Join-Path $logDir "server.stderr.log"
    try {
        Set-Content -LiteralPath $stdoutLog -Value "" -Encoding UTF8
        Set-Content -LiteralPath $stderrLog -Value "" -Encoding UTF8
    } catch [System.IO.IOException] {
        $runStamp = Get-Date -Format "yyyyMMdd-HHmmssfff"
        $stdoutLog = Join-Path $logDir ("server.$runStamp.stdout.log")
        $stderrLog = Join-Path $logDir ("server.$runStamp.stderr.log")
        Write-Host "The previous server log is in use; using a new log file for this run."
    }

    $arguments = @("-m", "uvicorn", "realtime_meeting.server:app", "--host", "127.0.0.1", "--port", "$Port")
    if ($Reload) { $arguments += "--reload" }
    $serverProcess = Start-Process `
        -FilePath $venvPython `
        -ArgumentList $arguments `
        -WorkingDirectory $projectRoot `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog `
        -WindowStyle Hidden `
        -PassThru

    $ready = $false
    $lastStatus = "waiting for service"
    for ($attempt = 0; $attempt -lt $maxWaitAttempts; $attempt++) {
        Start-Sleep -Milliseconds $pollIntervalMilliseconds
        try {
            $response = Invoke-RestMethod -Uri "$baseUrl/api/v2/health" -TimeoutSec 2
            if ($response.status) { $lastStatus = [string]$response.message }
            if ($response.status -eq "ready") { $ready = $true; break }
        } catch {
            if ($serverProcess.HasExited) {
                $details = ""
                if (Test-Path -LiteralPath $stderrLog) {
                    $details = (Get-Content -LiteralPath $stderrLog -Raw -ErrorAction SilentlyContinue).Trim()
                }
                if (-not $details) { $details = "No server error output was captured." }
                throw "The v2 server exited before it became ready (code $($serverProcess.ExitCode)). $details"
            }
        }
    }
    if (-not $ready) {
        if ($serverProcess.HasExited) {
            $details = (Get-Content -LiteralPath $stderrLog -Raw -ErrorAction SilentlyContinue).Trim()
            throw "The v2 server exited before its models became ready (code $($serverProcess.ExitCode)). $details"
        }
        throw "The v2 models did not become ready within $StartupTimeoutSeconds seconds (last status: $lastStatus). See $stderrLog"
    }

    Write-Host "Meeting v2 is running and models are ready at $baseUrl"
    Write-Host "Server logs: $stdoutLog and $stderrLog"
    if (-not $NoBrowser) { Start-Process $baseUrl | Out-Null }
    Wait-Process -Id $serverProcess.Id
} catch {
    Write-Host ("Startup failed: " + $_.Exception.Message) -ForegroundColor Red
    Write-Host "Press Enter to close this window." -ForegroundColor Yellow
    if ($Host.Name -match "ConsoleHost") { Read-Host | Out-Null }
    throw
} finally {
    if ($serverProcess -and -not $serverProcess.HasExited) {
        Stop-Process -Id $serverProcess.Id -Force
    }
    if ($ownsStartupMutex) {
        $startupMutex.ReleaseMutex()
    }
    $startupMutex.Dispose()
}
