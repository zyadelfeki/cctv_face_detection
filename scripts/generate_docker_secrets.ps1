# PowerShell script to generate Docker secrets
# Windows version of generate_docker_secrets.sh

Write-Host "=== Generating Docker Secrets ===" -ForegroundColor Green
Write-Host ""

# Create secrets directory
if (-not (Test-Path "secrets")) {
    New-Item -ItemType Directory -Path "secrets" | Out-Null
}

# Function to generate secure random password
function Generate-SecurePassword {
    param(
        [int]$Length = 25
    )
    
    # Use .NET crypto-secure random generator
    $bytes = New-Object byte[] $Length
    $rng = [System.Security.Cryptography.RNGCryptoServiceProvider]::Create()
    $rng.GetBytes($bytes)
    
    # Convert to base64 and clean up
    $password = [Convert]::ToBase64String($bytes)
    $password = $password -replace '[+/=]', ''
    return $password.Substring(0, $Length)
}

# Generate database password
$dbPassword = Generate-SecurePassword
$dbPassword | Out-File -FilePath "secrets/db_password.txt" -Encoding ASCII -NoNewline
Write-Host "✅ Generated database password: secrets/db_password.txt" -ForegroundColor Green

# Generate Grafana password
$grafanaPassword = Generate-SecurePassword
$grafanaPassword | Out-File -FilePath "secrets/grafana_password.txt" -Encoding ASCII -NoNewline
Write-Host "✅ Generated Grafana password: secrets/grafana_password.txt" -ForegroundColor Green

# Set read-only permissions
Set-ItemProperty -Path "secrets/db_password.txt" -Name IsReadOnly -Value $true
Set-ItemProperty -Path "secrets/grafana_password.txt" -Name IsReadOnly -Value $true

Write-Host ""
Write-Host "📝 Generated Credentials:" -ForegroundColor Cyan
Write-Host "   Database Password: $dbPassword"
Write-Host "   Grafana Password: $grafanaPassword"
Write-Host ""
Write-Host "⚠️  Save these credentials securely!" -ForegroundColor Yellow
Write-Host "🔒 Secret files have been created with read-only permissions" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "   1. Update .env.production with DATABASE_URL using this password"
Write-Host "   2. Run: docker-compose up -d"
Write-Host "   3. Access Grafana at http://localhost:3000 (admin / <grafana_password>)"
