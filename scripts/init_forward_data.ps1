$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path "$PSScriptRoot\..")

$researchDir = Resolve-Path "data\raw"
$forwardDir = Join-Path (Get-Location) "data\forward_raw"

if (-not (Test-Path $forwardDir)) {
    New-Item -ItemType Directory -Path $forwardDir | Out-Null
}

Write-Host "Initializing forward data from frozen research snapshot..."
Copy-Item -Path (Join-Path $researchDir "*") -Destination $forwardDir -Recurse -Force
Write-Host "Forward data initialized at $forwardDir"
Write-Host "Research data remains frozen at 2026-05-18."
