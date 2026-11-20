param(
    [string]$LanIPv4
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path -Parent $PSScriptRoot

if (-not $LanIPv4) {
    $LanIPv4 = Get-NetIPConfiguration |
        Where-Object { $_.IPv4DefaultGateway -and $_.IPv4Address } |
        ForEach-Object { $_.IPv4Address.IPAddress } |
        Where-Object { $_ -and $_ -notlike '127.*' -and $_ -notlike '169.254.*' } |
        Select-Object -First 1
}
if (-not $LanIPv4 -or $LanIPv4 -notmatch '^\d{1,3}(\.\d{1,3}){3}$') {
    throw 'Could not find a laptop LAN IPv4 address. Run with -LanIPv4 <address>.'
}

$backendPython = Join-Path $repositoryRoot 'backend\.venv\Scripts\python.exe'
$webDirectory = Join-Path $repositoryRoot 'web'
if (-not (Test-Path -LiteralPath $backendPython)) {
    throw "Backend virtual environment not found at $backendPython"
}
if (-not (Test-Path -LiteralPath (Join-Path $webDirectory 'node_modules'))) {
    throw 'Frontend dependencies are missing. Run npm install from web first.'
}
foreach ($port in @(8000, 5173)) {
    $probe = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $port)
    try {
        $probe.Start()
    } catch {
        throw "TCP port $port is already in use. Stop the existing service before starting the demo."
    } finally {
        $probe.Stop()
    }
}

$corsOrigins = "http://localhost:5173,http://127.0.0.1:5173,http://${LanIPv4}:5173"
$backendDirectory = Join-Path $repositoryRoot 'backend'
$backendCommand = @"
Set-Location -LiteralPath '$backendDirectory'
`$env:ATHLETELENS_CORS_ORIGINS = '$corsOrigins'
& '$backendPython' -m alembic upgrade head
if (`$LASTEXITCODE -ne 0) { throw 'Database migration failed; backend was not started.' }
& '$backendPython' -m uvicorn app.main:app --host 0.0.0.0 --port 8000
"@
$frontendCommand = @"
Set-Location -LiteralPath '$webDirectory'
`$env:VITE_API_BASE_URL = '/api'
`$env:VITE_API_PROXY_TARGET = 'http://127.0.0.1:8000'
& npm.cmd run dev -- --host 0.0.0.0 --port 5173 --strictPort
"@

$shell = (Get-Command powershell.exe -ErrorAction SilentlyContinue).Source
if (-not $shell) { $shell = (Get-Command pwsh.exe).Source }
$backendEncoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($backendCommand))
$frontendEncoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($frontendCommand))
Start-Process -FilePath $shell -ArgumentList @('-NoExit', '-EncodedCommand', $backendEncoded)
Start-Process -FilePath $shell -ArgumentList @('-NoExit', '-EncodedCommand', $frontendEncoded)

Write-Host ''
Write-Host "AthleteLens services are starting for LAN address $LanIPv4."
Write-Host "Phone URL:       http://${LanIPv4}:5173"
Write-Host "Backend health:  http://${LanIPv4}:8000/health"
Write-Host 'Keep both service windows open. Use a trusted private Wi-Fi network.'
