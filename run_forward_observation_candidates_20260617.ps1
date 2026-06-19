param(
    [Parameter(Mandatory=$true)]
    [string]$BaseAlpha,

    [Parameter(Mandatory=$true)]
    [string]$FullRerankAlpha,

    [Parameter(Mandatory=$true)]
    [string]$OutputRoot,

    [string]$MaxDataDate = "2026-05-18",
    [string]$Python = "",
    [string]$PortfolioValues = "500000,1000000",
    [switch]$SkipBacktest
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not $Python) {
    $candidatePython = Join-Path $env:USERPROFILE "miniconda3\envs\torch\python.exe"
    $Python = if ($env:PYTHON) { $env:PYTHON } elseif (Test-Path $candidatePython) { $candidatePython } else { "python" }
}

Write-Host "Forward observation candidates"
Write-Host "BaseAlpha=$BaseAlpha"
Write-Host "FullRerankAlpha=$FullRerankAlpha"
Write-Host "OutputRoot=$OutputRoot"
Write-Host "MaxDataDate=$MaxDataDate"

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$AlphaRoot = Join-Path $OutputRoot "alpha"
$BacktestRoot = Join-Path $OutputRoot "open_ledger"
New-Item -ItemType Directory -Force -Path $AlphaRoot | Out-Null
New-Item -ItemType Directory -Force -Path $BacktestRoot | Out-Null

$NegfilterAlpha = Join-Path $AlphaRoot "negfilter_r030_100_drop3.jsonl"
$EdgeAlpha = Join-Path $AlphaRoot "edge_r030_100.jsonl"

Write-Host "Building negfilter r030_100 drop3..."
& $Python run\make_negative_filter_from_full.py `
    --base-alpha $BaseAlpha `
    --full-rerank-alpha $FullRerankAlpha `
    --output-alpha $NegfilterAlpha `
    --start-rank 30 `
    --end-rank 100 `
    --drop-n 3

Write-Host "Building edge r030_100..."
& $Python run\make_edge_rerank_from_full.py `
    --base-alpha $BaseAlpha `
    --full-rerank-alpha $FullRerankAlpha `
    --output-alpha $EdgeAlpha `
    --start-rank 30 `
    --end-rank 100

if ($SkipBacktest) {
    Write-Host "SkipBacktest set. Alpha files generated only."
    exit 0
}

function Run-OpenLedger {
    param(
        [string]$Name,
        [string]$AlphaPath
    )
    $OutDir = Join-Path $BacktestRoot $Name
    Write-Host "Running open-ledger: $Name"
    & $Python run\backtest_retention_open_ledger.py `
        --alpha-jsonl $AlphaPath `
        --output-dir $OutDir `
        --target-fracs 0.006 `
        --hold-fracs 0.10 `
        --portfolio-values $PortfolioValues `
        --market-timing-mode legacy `
        --max-new-names 5 `
        --rebalance-band 0.20 `
        --max-data-date $MaxDataDate `
        --progress-every 3000
}

Run-OpenLedger -Name "main_candidate" -AlphaPath $BaseAlpha
Run-OpenLedger -Name "negfilter_r030_100_drop3" -AlphaPath $NegfilterAlpha
Run-OpenLedger -Name "edge_r030_100" -AlphaPath $EdgeAlpha

Write-Host "Done. Results written to $OutputRoot"
