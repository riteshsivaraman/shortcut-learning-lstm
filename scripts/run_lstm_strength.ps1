$configs = @(
    "configs/lstm_baseline.yaml",
    "configs/lstm_strength_25_end.yaml",
    "configs/lstm_strength_50_end.yaml",
    "configs/lstm_strength_75_end.yaml"
)

$seeds = @(42, 123, 7)

New-Item -ItemType Directory -Force -Path "logs" | Out-Null

$logFile = "logs/lstm_strength_sweep.log"

foreach ($config in $configs) {
    foreach ($seed in $seeds) {
        $msg = "=== $config seed=$seed ==="
        Write-Host $msg
        Add-Content -Path $logFile -Value $msg

        $result = python scripts/train.py --config $config --seed $seed 2>&1
        $result | Tee-Object -FilePath $logFile -Append

        if ($LASTEXITCODE -ne 0) {
            $failMsg = "FAILED: $config seed=$seed"
            Write-Host $failMsg -ForegroundColor Red
            Add-Content -Path $logFile -Value $failMsg
        }
    }
}

Add-Content -Path $logFile -Value "=== Sweep complete ==="
Write-Host "=== Sweep complete ===" -ForegroundColor Green
