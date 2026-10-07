$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$policyRun = 'runs/incoming-v2-final'
if (-not (Test-Path -LiteralPath (Join-Path $policyRun 'policy.zip'))) {
    $policyRun = 'models/precision-striker-v2'
}
$policyPython = 'python'
if (Test-Path -LiteralPath (Join-Path $env:USERPROFILE 'Anaconda3/python.exe')) {
    $policyPython = Join-Path $env:USERPROFILE 'Anaconda3/python.exe'
}
$manifest = Get-Content -LiteralPath (Join-Path $policyRun 'manifest.json') -Raw | ConvertFrom-Json
Write-Host "Completed PPO training: $($manifest.completed_timesteps) transitions, device $($manifest.actual_device)."
$evaluationPath = 'docs/experiment_data/incoming-v2-final-evaluation-200.json'
if (Test-Path -LiteralPath $evaluationPath) {
    $evaluation = Get-Content -LiteralPath $evaluationPath -Raw | ConvertFrom-Json
    Write-Host "Final test: $($evaluation.successes)/$($evaluation.episodes) goals; $($evaluation.collision_episodes) obstacle collisions."
    Write-Host "Design target met: $($evaluation.meets_report_target)"
}
Write-Host 'Opening the actual learned policy in Box2D. SPACE pauses; N resets; Q closes.'
Write-Host 'Episode outcomes will also appear here in the terminal.'
$scenarioPath = 'docs/experiment_data/example_contact_goal.json'
if (Test-Path -LiteralPath $scenarioPath) {
    Write-Host 'First: a selected successful contact/goal case to inspect centerline crossing, not typical performance.'
    Write-Host 'The final frame stays visible. Press N to try a new randomized scenario; A toggles auto-next.'
    & $policyPython -u -m scripts.watch_precision_striker --run $policyRun --scenario $scenarioPath
} else {
    & $policyPython -u -m scripts.watch_precision_striker --run $policyRun
}
if ($LASTEXITCODE -ne 0) { throw 'Viewer failed. See the error above.' }
