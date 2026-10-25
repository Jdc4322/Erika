$ErrorActionPreference = "Stop"
$api = "http://127.0.0.1:11434/api/tags"

try {
    Invoke-RestMethod -Uri $api -TimeoutSec 3 | Out-Null
    exit 0
} catch {
    # Start the local Ollama server if the desktop app has not started it.
}

$ollama = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
if (-not (Test-Path -LiteralPath $ollama)) {
    Write-Host "No encuentro Ollama en $ollama. Instala Ollama y vuelve a abrir Erika." -ForegroundColor Red
    exit 1
}

try {
    Start-Process -FilePath $ollama -ArgumentList "serve" -WindowStyle Hidden
} catch {
    Write-Host "No se pudo iniciar Ollama: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

for ($attempt = 0; $attempt -lt 30; $attempt++) {
    Start-Sleep -Seconds 1
    try {
        Invoke-RestMethod -Uri $api -TimeoutSec 2 | Out-Null
        exit 0
    } catch {
        # Keep waiting while Ollama starts.
    }
}

Write-Host "Ollama no respondió en http://127.0.0.1:11434. Comprueba que el servicio pueda iniciarse." -ForegroundColor Red
exit 1
