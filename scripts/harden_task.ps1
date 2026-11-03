# Scheduling hardening for the predlab paper loop. Run once, by hand, after reading it:
#   powershell -ExecutionPolicy Bypass -File scripts\harden_task.ps1
# 1. Lets the existing loop task wake the PC from sleep and run every 15 minutes
#    (it does not change sleep policy).
# 2. Registers a Friday 17:00 task that sends the weekly scorecard to Telegram.
# The weekly task only reads and sends. The hourly task also runs the executor in whatever
# mode executor/.env sets, so waking the PC means more of those cycles run.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Pyw = Join-Path $Root ".venv\Scripts\pythonw.exe"

# 1. Wake to run the hourly loop.
$task = Get-ScheduledTask -TaskName "predlab paper loop"
$task.Settings.WakeToRun = $true
$task.Triggers[0].Repetition.Interval = "PT15M"     # every 15 minutes (was hourly, then 30)
Set-ScheduledTask -InputObject $task | Out-Null
Write-Host "predlab paper loop: WakeToRun = true, every 15 minutes"

# 2. Weekly scorecard (Friday after the close).
$action = New-ScheduledTaskAction -Execute $Pyw -Argument "-m notify.cli weekly" -WorkingDirectory $Root
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Friday -At 5:00PM
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName "predlab weekly scorecard" -Action $action -Trigger $trigger `
    -Settings $settings -Description "Weekly paper-trading scorecard to Telegram" -Force | Out-Null
Write-Host "predlab weekly scorecard: registered (Fridays 17:00)"

# Wake timers must be allowed by the power plan for WakeToRun to work. Not changed here;
# to allow them on both AC and battery, run:
#   powercfg /setacvalueindex SCHEME_CURRENT SUB_SLEEP RTCWAKE 1
#   powercfg /setdcvalueindex SCHEME_CURRENT SUB_SLEEP RTCWAKE 1
#   powercfg /setactive SCHEME_CURRENT
Write-Host "Wake timers (current plan):"
powercfg /query SCHEME_CURRENT SUB_SLEEP RTCWAKE | Select-String "Current (AC|DC)"
