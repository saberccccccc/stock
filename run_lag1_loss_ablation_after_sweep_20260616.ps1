$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$candidatePython = Join-Path $env:USERPROFILE "miniconda3\envs\torch\python.exe"
$python = if ($env:PYTHON) { $env:PYTHON } elseif (Test-Path $candidatePython) { $candidatePython } else { "python" }
$waitPid = if ($env:WAIT_PID) { [int]$env:WAIT_PID } else { 0 }
if ($waitPid -gt 0) {
    while (Get-Process -Id $waitPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 30 }
}
& $python run\run_loss_ablation.py --config configs\lag1_loss_ablation_20260616.json
if ($LASTEXITCODE -ne 0) { throw "Lag1 loss ablation failed with exit code $LASTEXITCODE" }
