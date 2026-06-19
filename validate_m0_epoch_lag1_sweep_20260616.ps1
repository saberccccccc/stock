$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$candidatePython = Join-Path $env:USERPROFILE "miniconda3\envs\torch\python.exe"
$python = if ($env:PYTHON) { $env:PYTHON } elseif (Test-Path $candidatePython) { $candidatePython } else { "python" }
foreach ($epoch in 1..6) {
    $name = "m0_e$('{0:D3}' -f $epoch)"
    $ckpt = "checkpoints_loss_ablation_M0_nomulti\epochs\epoch_$('{0:D3}' -f $epoch).pt"
    & $python run\validate_candidate_models.py --output-dir lag1_checkpoint_sweep_m0_20260616 --checkpoint $ckpt --name $name --device cuda --progress-every 160
    if ($LASTEXITCODE -ne 0) { throw "Validation failed for $name with exit code $LASTEXITCODE" }
}
