param([int] $WaitForPid = 0)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $root
if ($WaitForPid -gt 0 -and (Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue)) {
    Wait-Process -Id $WaitForPid -Timeout 900
}
$python = Join-Path $root '.venv/Scripts/python.exe'
$output = Join-Path $root 'tmp/e27'
$project = Join-Path $root 'tmp/e24-work/project'
$status = Join-Path $output 'pipeline-exit.json'
"E27 batch started: $([DateTimeOffset]::Now.ToString('o'))"
$exitCode = 1
try {
    & $python 'mvp/blackbox_e27.py' --project $project --output $output --sync
    $exitCode = $LASTEXITCODE
} finally {
    & $python 'mvp/blackbox_e27_report.py' --input $output --export (Join-Path $root 'tmp/e27-report')
    $reportCode = $LASTEXITCODE
    $complete = $false
    $summary = Join-Path $root 'tmp/e27-report/stage-summary.json'
    if (($reportCode -eq 0) -and (Test-Path -LiteralPath $summary)) {
        $complete = (Get-Content -LiteralPath $summary -Raw | ConvertFrom-Json).complete
    }
    if (($exitCode -eq 0) -and $complete) {
        & $python 'mvp/blackbox_e27_report.py' --input $output --require-complete
        $reportCode = $LASTEXITCODE
    }
    @{ runnerExit = $exitCode; reportExit = $reportCode; complete = $complete; finished = [DateTimeOffset]::Now.ToString('o') } |
        ConvertTo-Json | Set-Content -LiteralPath $status -Encoding UTF8
}
if (($exitCode -ne 0) -or ($reportCode -ne 0)) { exit 1 }
