param(
    [Alias("ServerOnly")]
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    # `py -3.11` exits with an error when that minor version is absent. With
    # `$ErrorActionPreference = Stop`, probing a missing version aborts the
    # script before it can reach an installed 3.11 runtime. Read the launcher
    # inventory once instead, then select a supported interpreter explicitly.
    $pythonExe = $null
    $pyCommand = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $pyCommand) {
        $launcherInventory = @(& $pyCommand.Source -0p 2>$null)
        foreach ($line in $launcherInventory) {
            if ($line -match '^\s*-V:(3\.11)\s+(.+python\.exe)\s*$') {
                $pythonExe = $Matches[3].Trim()
                break
            }
        }
    }

    # Fall back to `python`/`python3` when the Python launcher is unavailable.
    if ($null -eq $pythonExe) {
        foreach ($commandName in @("python", "python3")) {
            $command = Get-Command $commandName -ErrorAction SilentlyContinue
            if ($null -eq $command) { continue }
            $version = (& $command.Source -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null).Trim()
            if ($version -match '^3\.11$') {
                $pythonExe = $command.Source
                break
            }
        }
    }

    if ($null -eq $pythonExe -or -not (Test-Path -LiteralPath $pythonExe)) {
        throw "Python 3.11 is required. Install Python 3.11 and try again."
    }
    & $pythonExe -m venv .venv
    & $venvPython -m pip install --upgrade pip setuptools wheel
    & $venvPython -m pip install "torch==2.11.0+cu128" "torchaudio==2.11.0+cu128" --index-url https://download.pytorch.org/whl/cu128
    & $venvPython -m pip install -e ".[dev]"
}

if (-not (Test-Path -LiteralPath (Join-Path $projectRoot ".env"))) {
    Write-Warning "Create .env from .env.example and add the rotated Jimo authorization value before generating minutes."
}

# Existing virtual environments may predate the web-only client. Install the
# project once more when the runtime dependencies are not available.
$dependencyCheck = & $venvPython -c "import ctranslate2, fastapi, faster_whisper, funasr, huggingface_hub, numpy, opencc, torch, websockets" 2>$null
if ($LASTEXITCODE -ne 0) {
    & $venvPython -m pip install -e ".[dev]"
}

# Fail early for an explicitly requested CUDA device instead of starting a
# backend that can only discover the problem after the model download begins.
$deviceCheck = & $venvPython -c "from realtime_meeting.config import load_settings; from realtime_meeting.runtime import choose_device; choose_device(load_settings().device)" 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Configured MEETING_DEVICE is unavailable: $($deviceCheck -join ' ')"
}

$arguments = @("-m", "realtime_meeting.cli")
if ($NoBrowser) {
    $arguments += "--no-browser"
} else {
    $arguments += "--browser"
}
& $venvPython @arguments
