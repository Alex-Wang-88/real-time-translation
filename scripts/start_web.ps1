param(
    [switch]$NoBrowser
)

$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location -LiteralPath $projectRoot

if ($NoBrowser) {
    & .\start.ps1 -NoBrowser
} else {
    & .\start.ps1
}
