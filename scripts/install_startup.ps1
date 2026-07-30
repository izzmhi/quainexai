<#
.SYNOPSIS
    Make Quainex start with Windows, with no console window.

.DESCRIPTION
    Registers two scheduled tasks that run at logon:

      Quainex Server  - the API, dashboard and Telegram bridge
      Quainex Tray    - the tray icon and the Ctrl+Alt+Q hotkey

    Why Task Scheduler rather than a Windows service:
        A service runs in session 0, isolated from your desktop. Quainex needs
        your desktop - it takes screenshots, reads the clipboard, launches
        applications and speaks through your speakers. A service could do none of
        that. Task Scheduler at logon runs in your session, as you, which is
        exactly the access Quainex is designed around.

        It also means no elevation. Everything here is per-user, installs without
        an administrator prompt, and cannot affect anyone else on the machine -
        which matches where the credential vault stores keys.

    Why pythonw.exe:
        `python.exe` owns a console window. Started at logon that is a black box
        on screen forever, and closing it kills Quainex. `pythonw.exe` has no
        console, so the server lives in the background like anything else that
        starts with Windows. Logs go to `logs\quainex.log` as usual, which is
        where you look instead of at a terminal.

    NOTE: ASCII-only. Windows PowerShell 5.1 reads a BOM-less script as ANSI, and
    one non-ASCII character corrupts the parse.

.PARAMETER Remove
    Uninstall both tasks instead of installing them.

.PARAMETER ServerOnly
    Install the server task but not the tray icon.

.EXAMPLE
    .\scripts\install_startup.ps1

.EXAMPLE
    .\scripts\install_startup.ps1 -Remove
#>
[CmdletBinding()]
param(
    [switch]$Remove,
    [switch]$ServerOnly
)

$ErrorActionPreference = "Stop"

$serverTask = "Quainex Server"
$trayTask = "Quainex Tray"

function Remove-QuainexTask {
    param([string]$Name)
    $existing = Get-ScheduledTask -TaskName $Name -ErrorAction SilentlyContinue
    if ($existing) {
        Unregister-ScheduledTask -TaskName $Name -Confirm:$false
        Write-Host "  removed: $Name" -ForegroundColor DarkGray
        return $true
    }
    return $false
}

if ($Remove) {
    Write-Host "Removing Quainex startup tasks" -ForegroundColor Cyan
    $any = $false
    foreach ($name in @($serverTask, $trayTask)) {
        if (Remove-QuainexTask -Name $name) { $any = $true }
    }
    if (-not $any) { Write-Host "  nothing was installed" -ForegroundColor DarkGray }
    Write-Host "Done. Quainex will no longer start with Windows." -ForegroundColor Green
    exit 0
}

# --- locate the interpreter and the project -------------------------------

$repo = Split-Path -Parent $PSScriptRoot
$pythonw = Join-Path $repo ".venv\Scripts\pythonw.exe"
$entry = Join-Path $repo "main.py"

Write-Host "Installing Quainex startup tasks" -ForegroundColor Cyan
Write-Host "  project: $repo" -ForegroundColor DarkGray

if (-not (Test-Path $pythonw)) {
    throw "No virtual environment found at $pythonw. Create one first: py -3.12 -m venv .venv"
}
if (-not (Test-Path $entry)) {
    throw "main.py not found at $entry. Run this script from inside the Quainex repo."
}

# --- the server -----------------------------------------------------------

Remove-QuainexTask -Name $serverTask | Out-Null

$serverAction = New-ScheduledTaskAction -Execute $pythonw -Argument "`"$entry`"" -WorkingDirectory $repo
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME

# RestartCount/RestartInterval rather than a supervisor process: if Quainex dies,
# Windows brings it back. StartWhenAvailable covers a logon where the network is
# not up yet, which matters because the Telegram bridge wants to reach out.
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Seconds 0)

Register-ScheduledTask `
    -TaskName $serverTask `
    -Action $serverAction `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Quainex AI operating system: API, dashboard and phone bridge." | Out-Null

Write-Host "  installed: $serverTask" -ForegroundColor Green

# --- the tray -------------------------------------------------------------

if (-not $ServerOnly) {
    Remove-QuainexTask -Name $trayTask | Out-Null

    $trayAction = New-ScheduledTaskAction -Execute $pythonw `
        -Argument "-m quainex.desktop.tray" -WorkingDirectory $repo

    Register-ScheduledTask `
        -TaskName $trayTask `
        -Action $trayAction `
        -Trigger $trigger `
        -Settings $settings `
        -Description "Quainex tray icon and Ctrl+Alt+Q global hotkey." | Out-Null

    Write-Host "  installed: $trayTask" -ForegroundColor Green
}

# --- start now, so it is not a mystery until the next reboot --------------

Write-Host "`nStarting now" -ForegroundColor Cyan
Start-ScheduledTask -TaskName $serverTask
if (-not $ServerOnly) { Start-ScheduledTask -TaskName $trayTask }

# Poll rather than sleep-and-hope: report what is actually true.
$up = $false
foreach ($attempt in 1..20) {
    Start-Sleep -Milliseconds 1000
    try {
        $health = Invoke-RestMethod "http://127.0.0.1:8000/health" -TimeoutSec 2
        $up = $true
        Write-Host "  $($health.app) $($health.version) is up" -ForegroundColor Green
        break
    } catch {
        # Still starting.
    }
}

if (-not $up) {
    Write-Host "  the server did not answer within 20s - check logs\quainex.log" -ForegroundColor Yellow
}

Write-Host @"

Quainex now starts when you log in.

  Console       http://127.0.0.1:8000/ui/
  Hotkey        Ctrl+Alt+Q
  Tray          click the icon in the notification area
  Logs          logs\quainex.log
  Uninstall     .\scripts\install_startup.ps1 -Remove

"@ -ForegroundColor Cyan
