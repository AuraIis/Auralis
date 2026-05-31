param(
    [Parameter(Mandatory = $true)]
    [string]$Command,

    [double]$MaxPrivateMemoryGB = 24,
    [double]$MinFreeCommitGB = 12,
    [int]$PollSeconds = 5,
    [string]$WorkingDirectory = "."
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Get-CommitInfoGB {
    $os = Get-CimInstance Win32_OperatingSystem
    $total = [double]$os.TotalVirtualMemorySize / 1MB
    $free = [double]$os.FreeVirtualMemory / 1MB
    [PSCustomObject]@{
        TotalGB = $total
        FreeGB = $free
        UsedGB = $total - $free
    }
}

Write-Host "[guard] command: $Command"
Write-Host "[guard] max private memory: $MaxPrivateMemoryGB GB"
Write-Host "[guard] min free commit: $MinFreeCommitGB GB"

$resolvedWorkingDirectory = (Resolve-Path -LiteralPath $WorkingDirectory).Path
$safeWorkingDirectory = $resolvedWorkingDirectory.Replace("'", "''")
$guardedCommand = "Set-Location -LiteralPath '$safeWorkingDirectory'; $Command"
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = "powershell.exe"
$encodedCommand = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($guardedCommand))
$psi.Arguments = "-NoProfile -ExecutionPolicy Bypass -EncodedCommand $encodedCommand"
$psi.WorkingDirectory = $env:TEMP
$psi.UseShellExecute = $false

$proc = [System.Diagnostics.Process]::Start($psi)
Write-Host "[guard] started pid=$($proc.Id)"

try {
    while (-not $proc.HasExited) {
        Start-Sleep -Seconds $PollSeconds
        $proc.Refresh()
        $privateGB = [double]$proc.PrivateMemorySize64 / 1GB
        $commit = Get-CommitInfoGB
        Write-Host ("[guard] pid={0} private={1:N1}GB commit_used={2:N1}/{3:N1}GB free={4:N1}GB" -f `
            $proc.Id, $privateGB, $commit.UsedGB, $commit.TotalGB, $commit.FreeGB)

        if ($privateGB -gt $MaxPrivateMemoryGB) {
            Write-Error ("[guard] killing pid={0}: private memory {1:N1}GB > limit {2:N1}GB" -f `
                $proc.Id, $privateGB, $MaxPrivateMemoryGB)
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            exit 91
        }
        if ($commit.FreeGB -lt $MinFreeCommitGB) {
            Write-Error ("[guard] killing pid={0}: free commit {1:N1}GB < limit {2:N1}GB" -f `
                $proc.Id, $commit.FreeGB, $MinFreeCommitGB)
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            exit 92
        }
    }
}
finally {
    if (-not $proc.HasExited) {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "[guard] process exited with code $($proc.ExitCode)"
exit $proc.ExitCode
