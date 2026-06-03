$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
  throw "Missing venv. Run setup.ps1 first."
}
Set-Location $root
$bindHost = if ($env:SEEDANCE_HOST) { $env:SEEDANCE_HOST } else { "0.0.0.0" }
$port = if ($env:SEEDANCE_PORT) { $env:SEEDANCE_PORT } else { "18080" }
$reloadArgs = if ($env:SEEDANCE_RELOAD -eq "1") { @("--reload") } else { @() }
& $python -m uvicorn app.backend.main:app --host $bindHost --port $port @reloadArgs
