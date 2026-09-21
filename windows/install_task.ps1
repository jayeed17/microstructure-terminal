# Registers one always-on collector per symbol. Survives reboots and Windows
# Update, runs without anyone logged in.
#
# From the repo root, in an ADMIN PowerShell:
#   powershell -ExecutionPolicy Bypass -File windows\install_task.ps1
#   powershell -ExecutionPolicy Bypass -File windows\install_task.ps1 -Symbols btcusdt,ethusdt,solusdt
#   powershell -ExecutionPolicy Bypass -File windows\install_task.ps1 -Uninstall

param(
  [string[]]$Symbols = @("btcusdt", "ethusdt"),
  [switch]$Uninstall
)

$repo = Split-Path -Parent $PSScriptRoot
$bat  = Join-Path $repo "windows\run_collector.bat"
$user = "$env:USERDOMAIN\$env:USERNAME"

if (-not (Test-Path (Join-Path $repo ".venv\Scripts\python.exe"))) {
  Write-Error "No .venv found in $repo. Create it first: py -m venv .venv; .venv\Scripts\pip install -r requirements.txt"
  exit 1
}

foreach ($s in $Symbols) {
  $name = "microstructure-collector-$s"

  if ($Uninstall) {
    Stop-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
      Where-Object { $_.CommandLine -match "collect.py --symbol $s" } |
      ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
    Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "removed $name"
    continue
  }

  $action    = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$bat`" $s" -WorkingDirectory $repo
  $trigger   = New-ScheduledTaskTrigger -AtStartup
  # S4U = runs without you logged in, no stored password; still has internet
  $principal = New-ScheduledTaskPrincipal -UserId $user -LogonType S4U -RunLevel Limited
  $settings  = New-ScheduledTaskSettingsSet `
                 -ExecutionTimeLimit ([TimeSpan]::Zero) `
                 -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
                 -StartWhenAvailable -MultipleInstances IgnoreNew `
                 -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

  Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings -Force | Out-Null
  Start-ScheduledTask -TaskName $name
  Write-Host "registered + started $name"
}

if (-not $Uninstall) {
  powercfg /change standby-timeout-ac 0
  powercfg /change hibernate-timeout-ac 0
  Write-Host "sleep/hibernate disabled on AC power"
}
