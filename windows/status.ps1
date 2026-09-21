# Quick health check. Run from repo root:  powershell -File windows\status.ps1
$repo = Split-Path -Parent $PSScriptRoot
Get-ScheduledTask -TaskName "microstructure-collector-*" -ErrorAction SilentlyContinue |
  Select-Object TaskName, State | Format-Table -AutoSize

$files = Get-ChildItem (Join-Path $repo "data\raw_*.parquet") -ErrorAction SilentlyContinue
if ($files) {
  $mb = [math]::Round(($files | Measure-Object Length -Sum).Sum / 1MB, 1)
  $last = ($files | Sort-Object LastWriteTime | Select-Object -Last 1)
  Write-Host "chunks: $($files.Count)   size: $mb MB   last write: $($last.LastWriteTime)"
} else { Write-Host "no data chunks yet (first flush after ~5000 ticks, ~8 min)" }

Get-ChildItem (Join-Path $repo "logs\*.log") -ErrorAction SilentlyContinue | ForEach-Object {
  Write-Host "`n--- $($_.Name) ---"; Get-Content $_.FullName -Tail 4
}
