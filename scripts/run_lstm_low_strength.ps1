$ErrorActionPreference = "Stop"

$configs = @(
    "configs/lstm_strength_2_end.yaml",
    "configs/lstm_strength_5_end.yaml",
    "configs/lstm_strength_10_end.yaml"
)

$seeds = @(42, 123, 7)
$outputCsv = "results/all_runs_lstm_low_strength.csv"

New-Item -ItemType Directory -Force -Path logs | Out-Null
$failedLog = "logs/failed_runs_low_strength.txt"
if (Test-Path $failedLog) { Remove-Item $failedLog }

foreach ($config in $configs) {
    foreach ($seed in $seeds) {
        $timestamp = Get-Date -Format "HH:mm:ss"
        Write-Host "=== $timestamp $config seed=$seed ===" -ForegroundColor Cyan
        try {
            python scripts/train.py --config $config --seed $seed --output-csv $outputCsv
        } catch {
            $msg = "FAILED: $config seed=$seed"
            Write-Host $msg -ForegroundColor Red
            Add-Content $failedLog $msg
        }
    }
}

$timestamp = Get-Date -Format "HH:mm:ss"
Write-Host "=== Sweep complete at $timestamp ===" -ForegroundColor Green

if (Test-Path $failedLog) {
    Write-Host "`nFailed runs:" -ForegroundColor Yellow
    Get-Content $failedLog
    Write-Host "`nSkipping issue close due to failed runs." -ForegroundColor Yellow
} else {
    Write-Host "All runs succeeded. Results saved to $outputCsv" -ForegroundColor Green
    gh issue close 36 --comment "All 9 runs complete (lstm_strength_2_end, lstm_strength_5_end, lstm_strength_10_end x seeds 42, 123, 7). Results saved to results/all_runs_lstm_low_strength.csv."
    Write-Host "GitHub issue #36 closed." -ForegroundColor Green
}
