$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path "$PSScriptRoot\..")

$candidatePython = Join-Path $env:USERPROFILE "miniconda3\envs\torch\python.exe"
$python = if ($env:PYTHON) {
    $env:PYTHON
} elseif (Test-Path $candidatePython) {
    $candidatePython
} else {
    "python"
}

& $python run\train.py --model v9 --epochs 25 --device cuda --output-dir checkpoints_exp
if ($LASTEXITCODE -ne 0) {
    throw "V9 training failed with exit code $LASTEXITCODE"
}

& $python run\train.py --model gat --epochs 25 --device cuda --output-dir checkpoints_exp
if ($LASTEXITCODE -ne 0) {
    throw "GAT training failed with exit code $LASTEXITCODE"
}
