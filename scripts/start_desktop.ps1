param(
    [switch]$ServerOnly
)

$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location -LiteralPath $projectRoot

if ($ServerOnly) {
    & .\start.ps1 -ServerOnly
} else {
    & .\start.ps1
}
