$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path "$PSScriptRoot\..")

$python = "C:\Users\x\miniconda3\envs\torch\python"

& $python run\train.py --model v9 --epochs 25 --device cuda --output-dir checkpoints_exp
if ($LASTEXITCODE -ne 0) {
    throw "V9 training failed with exit code $LASTEXITCODE"
}

& $python run\train.py --model gat --epochs 25 --device cuda --output-dir checkpoints_exp
if ($LASTEXITCODE -ne 0) {
    throw "GAT training failed with exit code $LASTEXITCODE"
}
