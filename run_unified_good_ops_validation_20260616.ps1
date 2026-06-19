$ErrorActionPreference = "Stop"

$candidatePython = Join-Path $env:USERPROFILE "miniconda3\envs\torch\python.exe"
$python = if ($env:PYTHON) { $env:PYTHON } elseif (Test-Path $candidatePython) { $candidatePython } else { "python" }
Set-Location $PSScriptRoot

$outRoot = "unified_good_ops_validation_20260616"
New-Item -ItemType Directory -Force -Path $outRoot | Out-Null

$candidates = @(
    @{ Name = "frozen_v9"; Alpha = "lag1_checkpoint_sweep_m0_20260616\frozen_v9\alpha_maxret095.jsonl" },
    @{ Name = "m0_e006"; Alpha = "lag1_checkpoint_sweep_m0_20260616\m0_e006\alpha_maxret095.jsonl" },
    @{ Name = "a4_e6"; Alpha = "candidate_model_validation_20260614\a4_e6\alpha_maxret095.jsonl" },
    @{ Name = "v3"; Alpha = "reranker_validation_20260615\regression_v3\alpha_maxret095.jsonl" },
    @{ Name = "v4"; Alpha = "reranker_validation_20260615\gated_v4\alpha_maxret095.jsonl" }
)

$shareScenarios = @(
    @{ Name = "base"; AdvCap = "0.05"; CostMult = 1.0; Lag = "0" },
    @{ Name = "cap3"; AdvCap = "0.03"; CostMult = 1.0; Lag = "0" },
    @{ Name = "cost2x"; AdvCap = "0.05"; CostMult = 2.0; Lag = "0" },
    @{ Name = "lag1"; AdvCap = "0.05"; CostMult = 1.0; Lag = "1" }
)

$openMarketScenarios = @(
    @{ Name = "legacy"; Mode = "legacy"; MinMult = "0.20" },
    @{ Name = "dynamic_m20"; Mode = "dynamic"; MinMult = "0.20" },
    @{ Name = "dynamic_m30"; Mode = "dynamic"; MinMult = "0.30" }
)

function Invoke-Checked {
    param(
        [string]$Label,
        [string[]]$CmdArgs
    )
    Write-Host "=== $Label ==="
    & $python @CmdArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed for $Label with exit code $LASTEXITCODE"
    }
}

foreach ($candidate in $candidates) {
    foreach ($scenario in $shareScenarios) {
        foreach ($portfolio in @("500000", "1000000")) {
            $costMult = [double]$scenario.CostMult
            $commission = (0.0001 * $costMult).ToString("0.########", [Globalization.CultureInfo]::InvariantCulture)
            $stamp = (0.0005 * $costMult).ToString("0.########", [Globalization.CultureInfo]::InvariantCulture)
            $slippage = (0.0005 * $costMult).ToString("0.########", [Globalization.CultureInfo]::InvariantCulture)
            $outDir = Join-Path $outRoot ("share_top30\" + $candidate.Name + "\" + $scenario.Name + "_" + ([int]([double]$portfolio / 10000)) + "w")
            Invoke-Checked "share_top30 $($candidate.Name) $($scenario.Name) $portfolio" @(
                "run\backtest_retention_execution_constraints.py",
                "--alpha-jsonl", $candidate.Alpha,
                "--data-dir", "data\raw",
                "--output-dir", $outDir,
                "--target-fracs", "0.006",
                "--hold-fracs", "0.10",
                "--portfolio-value", $portfolio,
                "--adv-participation-cap", $scenario.AdvCap,
                "--min-adv-cny", "3000000",
                "--rebalance-band", "0.20",
                "--market-timing-mode", "legacy",
                "--limit-threshold", "0.095",
                "--execution-lag", $scenario.Lag,
                "--commission-rate", $commission,
                "--stamp-tax-rate", $stamp,
                "--slippage-rate", $slippage,
                "--lot-size", "100",
                "--min-commission-cny", "5",
                "--progress-every", "2000"
            )
        }
    }
}

foreach ($portfolio in @("500000", "1000000")) {
    $outDir = Join-Path $outRoot ("share_fallback_top20\frozen_v9_base_" + ([int]([double]$portfolio / 10000)) + "w")
    Invoke-Checked "share_fallback_top20 frozen_v9 base $portfolio" @(
        "run\backtest_retention_execution_constraints.py",
        "--alpha-jsonl", "lag1_checkpoint_sweep_m0_20260616\frozen_v9\alpha_maxret095.jsonl",
        "--data-dir", "data\raw",
        "--output-dir", $outDir,
        "--target-fracs", "0.004",
        "--hold-fracs", "0.06",
        "--portfolio-value", $portfolio,
        "--adv-participation-cap", "0.05",
        "--min-adv-cny", "3000000",
        "--rebalance-band", "0.20",
        "--market-timing-mode", "legacy",
        "--limit-threshold", "0.095",
        "--execution-lag", "0",
        "--lot-size", "100",
        "--min-commission-cny", "5",
        "--progress-every", "2000"
    )
}

foreach ($candidate in $candidates) {
    foreach ($market in $openMarketScenarios) {
        foreach ($portfolio in @("500000", "1000000")) {
            $outDir = Join-Path $outRoot ("open_to_open\" + $candidate.Name + "\" + $market.Name + "_" + ([int]([double]$portfolio / 10000)) + "w")
            Invoke-Checked "open_to_open $($candidate.Name) $($market.Name) $portfolio" @(
                "run\backtest_retention_open_execution.py",
                "--alpha-jsonl", $candidate.Alpha,
                "--data-dir", "data\raw",
                "--output-dir", $outDir,
                "--target-fracs", "0.03,0.035",
                "--hold-fracs", "0.50,0.60",
                "--return-mode", "open_to_open",
                "--portfolio-value", $portfolio,
                "--adv-participation-cap", "0.05",
                "--min-adv-cny", "20000000",
                "--market-timing-mode", $market.Mode,
                "--market-min-mult", $market.MinMult,
                "--market-max-mult", "1.00",
                "--limit-threshold", "0.095",
                "--execution-lag", "0",
                "--progress-every", "2000"
            )
        }
    }
}

Write-Host "Unified good-ops validation finished."
