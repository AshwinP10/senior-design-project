param(
    [Parameter(Mandatory=$true)][string]$Run,
    [int]$Obstacles = 1
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$replayPath = Join-Path $Run ('terminal-replay-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
Write-Host 'Actual saved PPO policy playback. A completed training run does not imply scoring success.'
$policyPython = 'python'
if (Test-Path -LiteralPath (Join-Path $env:USERPROFILE 'Anaconda3/python.exe')) {
    $policyPython = Join-Path $env:USERPROFILE 'Anaconda3/python.exe'
}
& $policyPython -u -m scripts.eval_precision_striker --run $Run --output $replayPath --layouts 1 --shots 1 --obstacles $Obstacles --terminal
if ($LASTEXITCODE -ne 0) { throw 'Policy replay failed. See the error above.' }
Write-Host "Replay and outcomes saved to $replayPath"
