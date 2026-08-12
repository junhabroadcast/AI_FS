# AI FS Monitor exe 빌드 (WFM tools/build.ps1 스타일)
$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot

Push-Location $Root
try {
    pyinstaller --noconfirm --clean --onedir --windowed `
        --name AiFsMonitor `
        --collect-submodules comtypes `
        ai_fs_monitor.py
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }
}
finally {
    Pop-Location
}

Write-Host "Built: $Root\dist\AiFsMonitor\AiFsMonitor.exe"
