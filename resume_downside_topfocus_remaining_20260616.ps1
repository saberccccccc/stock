$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$candidatePython = Join-Path $env:USERPROFILE "miniconda3\envs\torch\python.exe"
$python = if ($env:PYTHON) { $env:PYTHON } elseif (Test-Path $candidatePython) { $candidatePython } else { "python" }

& $python run\train.py --model v9 --epochs 6 --lr 0.0001 --device cuda --batch-size 4 --val-batch-size 2 --accum-steps 4 --memmap-trim-interval 32 --best-val-metric rawtopstable_h5_top0p6 --eval-top-fracs 0.006,0.01,0.02,0.05,0.10 --train-label-end 2023-12-31 --val-label-end 2024-12-31 --early-stop-patience 6 --seed 42 --save-every-epoch --horizon-weights 0.15,0.25,0.35,0.25 --industry-loss-weight 0.0 --multi-loss-weight 0.0 --diversity-loss-weight 0.0 --spread-loss-weight 0.0 --top-focus-temperature 0.75 --top-focus-delay-epochs 2 --downside-temperature 0.75 --downside-delay-epochs 2 --output-dir checkpoints_loss_ablation_T001 --downside-loss-weight 0.0 --top-focus-loss-weight 0.001 --resume-from checkpoints_loss_ablation_T001\epochs\epoch_004.pt --reset-optimizer
if ($LASTEXITCODE -ne 0) { throw "T001 resume failed with exit code $LASTEXITCODE" }

& $python run\train.py --model v9 --epochs 6 --lr 0.0001 --device cuda --batch-size 4 --val-batch-size 2 --accum-steps 4 --memmap-trim-interval 32 --best-val-metric rawtopstable_h5_top0p6 --eval-top-fracs 0.006,0.01,0.02,0.05,0.10 --train-label-end 2023-12-31 --val-label-end 2024-12-31 --early-stop-patience 6 --seed 42 --save-every-epoch --horizon-weights 0.15,0.25,0.35,0.25 --industry-loss-weight 0.0 --multi-loss-weight 0.0 --diversity-loss-weight 0.0 --spread-loss-weight 0.0 --top-focus-temperature 0.75 --top-focus-delay-epochs 2 --downside-temperature 0.75 --downside-delay-epochs 2 --output-dir checkpoints_loss_ablation_D003_T001 --downside-loss-weight 0.003 --top-focus-loss-weight 0.001
if ($LASTEXITCODE -ne 0) { throw "D003_T001 failed with exit code $LASTEXITCODE" }
