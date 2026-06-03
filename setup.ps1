$ErrorActionPreference = "Stop"

$root = $PSScriptRoot
$venv = Join-Path $root ".venv"
$tmp = Join-Path $root "tmp"
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

$pythonCandidates = @(
  "C:\Users\yibo\miniforge3\envs\lerobot\python.exe",
  "py"
)

$python = $null
foreach ($candidate in $pythonCandidates) {
  if ($candidate -eq "py") {
    if (Get-Command py -ErrorAction SilentlyContinue) { $python = "py"; break }
  } elseif (Test-Path -LiteralPath $candidate) {
    $python = $candidate
    break
  }
}
if (-not $python) { throw "No Python found." }

if (-not (Test-Path -LiteralPath $venv)) {
  if ($python -eq "py") {
    & py -3 -m venv $venv
  } else {
    & $python -m venv $venv
  }
}

$env:TMP = $tmp
$env:TEMP = $tmp
$env:PIP_CACHE_DIR = $tmp
$pip = Join-Path $venv "Scripts\python.exe"
& $pip -m pip install --upgrade pip
& $pip -m pip install --no-cache-dir -r (Join-Path $root "requirements.txt")
& $pip -m pip install --no-cache-dir (Join-Path $root "vendor\wheels\nmx_msg-2.2.0-py3-none-any.whl")

Write-Host "Setup complete: $venv"
