$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
  throw "Missing venv. Run setup.ps1 first."
}
Set-Location $root
& $python -m unittest discover -s tests -p "test_*.py"
