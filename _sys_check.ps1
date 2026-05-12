Write-Host "=== CPU Usage ==="
$cpu = Get-CimInstance Win32_Processor
Write-Host "CPU: $($cpu.Name.Trim())"
Write-Host "Cores: $($cpu.NumberOfCores) | Threads: $($cpu.NumberOfLogicalProcessors)"
Write-Host ""

Write-Host "=== Memory Usage ==="
$os = Get-CimInstance Win32_OperatingSystem
$total = [math]::Round($os.TotalVisibleMemorySize/1MB, 1)
$free = [math]::Round($os.FreePhysicalMemory/1MB, 1)
$used = [math]::Round(($os.TotalVisibleMemorySize - $os.FreePhysicalMemory)/1MB, 1)
Write-Host "Total: ${total}GB | Used: ${used}GB | Free: ${free}GB"
Write-Host ""

Write-Host "=== Top CPU Processes ==="
Get-Process | Sort-Object CPU -Descending | Select-Object -First 10 | ForEach-Object {
    $mem = [math]::Round($_.WorkingSet64/1MB, 1)
    Write-Host "$($_.Name.PadRight(25)) CPU: $([math]::Round($_.CPU, 1))s | Mem: ${mem}MB"
}
Write-Host ""

Write-Host "=== Python Process ==="
Get-Process python -ErrorAction SilentlyContinue | ForEach-Object {
    $mem = [math]::Round($_.WorkingSet64/1MB, 1)
    Write-Host "PID: $($_.Id) | Threads: $($_.Threads.Count) | CPU_time: $([math]::Round($_.CPU, 1))s | Memory: ${mem}MB"
}
Write-Host ""

Write-Host "=== GPU ==="
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv
