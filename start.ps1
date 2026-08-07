param(
    [Alias("NoBrowser")]
    [switch]$ServerOnly
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    # `py -3.12` exits with an error when that minor version is absent. With
    # `$ErrorActionPreference = Stop`, probing a missing version aborts the
    # script before it can reach an installed 3.11 runtime. Read the launcher
    # inventory once instead, then select a supported interpreter explicitly.
    $pythonExe = $null
    $pyCommand = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $pyCommand) {
        $launcherInventory = @(& $pyCommand.Source -0p 2>$null)
        foreach ($line in $launcherInventory) {
            if ($line -match '^\s*-V:(3\.(10|11|12))\s+(.+python\.exe)\s*$') {
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
            if ($version -match '^3\.(10|11|12)$') {
                $pythonExe = $command.Source
                break
            }
        }
    }

    if ($null -eq $pythonExe -or -not (Test-Path -LiteralPath $pythonExe)) {
        throw "Python 3.10-3.12 is required. Install Python 3.11 or 3.12 and try again."
    }
    & $pythonExe -m venv .venv
    & $venvPython -m pip install --upgrade pip setuptools wheel
    & $venvPython -m pip install torch --index-url https://download.pytorch.org/whl/cu128
    & $venvPython -m pip install -e ".[dev]"
}

if (-not (Test-Path -LiteralPath (Join-Path $projectRoot ".env"))) {
    Write-Warning "Create .env from .env.example and add the rotated Jimo authorization value before generating minutes."
}

# Existing virtual environments may predate the desktop client. Install the
# project once more when the new GUI/audio dependencies are not available.
$dependencyCheck = & $venvPython -c "import PyQt6, sounddevice, opencc" 2>$null
if ($LASTEXITCODE -ne 0) {
    & $venvPython -m pip install -e ".[dev]"
}

$arguments = if ($ServerOnly) {
    @("-m", "realtime_meeting.cli", "--no-browser")
} else {
    @("-m", "realtime_meeting.desktop")
}
& $venvPython @arguments
