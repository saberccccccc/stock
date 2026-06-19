$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$candidatePython = Join-Path $env:USERPROFILE "miniconda3\envs\torch\python.exe"
$python = if ($env:PYTHON) { $env:PYTHON } elseif (Test-Path $candidatePython) { $candidatePython } else { "python" }
$candidates = @("m0_nomulti_e6", "d001_e6", "d003_e6", "d005_e6", "t001_e5", "d003_t001_e6")
foreach ($candidate in $candidates) {
    & $python run\validate_candidate_models.py --output-dir downside_topfocus_validation_20260616 --only $candidate --device cuda --progress-every 120
    if ($LASTEXITCODE -ne 0) { throw "Validation failed for $candidate with exit code $LASTEXITCODE" }
}
