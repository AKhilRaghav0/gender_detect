# Winget Installation Script for Windows 11
# This script will install winget if it's not already available

Write-Host "🛠️ Installing Winget (Windows Package Manager)" -ForegroundColor Green
Write-Host "=" * 50 -ForegroundColor Yellow

# Check if winget is already installed
try {
    $wingetVersion = winget --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Winget is already installed!" -ForegroundColor Green
        Write-Host "Version: $wingetVersion" -ForegroundColor Cyan
        exit 0
    }
} catch {
    Write-Host "⚠️ Winget not found, proceeding with installation..." -ForegroundColor Yellow
}

# Method 1: Try installing via Microsoft Store
Write-Host "📦 Method 1: Installing via Microsoft Store..." -ForegroundColor Cyan
try {
    # Try to install App Installer (which includes winget)
    Start-Process "ms-windows-store://pdp/?productid=9NBLGGH4NNS1" -Wait
    Write-Host "✅ Microsoft Store opened for App Installer" -ForegroundColor Green
    Write-Host "📋 Please install 'App Installer' from the Microsoft Store" -ForegroundColor Yellow
    Read-Host "Press Enter after installing App Installer from Microsoft Store"
} catch {
    Write-Host "❌ Microsoft Store method failed" -ForegroundColor Red
}

# Method 2: Manual download and installation
Write-Host "⬇️ Method 2: Downloading latest winget release..." -ForegroundColor Cyan

$downloadUrl = "https://github.com/microsoft/winget-cli/releases/latest/download/Microsoft.DesktopAppInstaller_8wekyb3d8bbwe.msixbundle"
$installerPath = "$env:TEMP\Microsoft.DesktopAppInstaller.msixbundle"

try {
    Write-Host "Downloading winget installer..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri $downloadUrl -OutFile $installerPath -UseBasicParsing

    Write-Host "Installing winget..." -ForegroundColor Yellow
    Add-AppxPackage -Path $installerPath

    Write-Host "✅ Winget installation completed!" -ForegroundColor Green
    Write-Host "🔄 Please restart your PowerShell/Command Prompt" -ForegroundColor Yellow

    # Test installation
    Start-Sleep -Seconds 3
    try {
        $testVersion = winget --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "🎉 Winget is working! Version: $testVersion" -ForegroundColor Green
        }
    } catch {
        Write-Host "⚠️ Winget installed but may need system restart" -ForegroundColor Yellow
    }

} catch {
    Write-Host "❌ Manual installation failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "" -ForegroundColor Red
    Write-Host "🔄 Alternative installation methods:" -ForegroundColor Yellow
    Write-Host "1. Open Microsoft Store → Search 'App Installer' → Install" -ForegroundColor White
    Write-Host "2. Visit: https://github.com/microsoft/winget-cli/releases" -ForegroundColor White
    Write-Host "3. Download and install the latest .msixbundle" -ForegroundColor White
    Write-Host "4. After installation, restart your computer" -ForegroundColor White
}

# Final verification
Write-Host "" -ForegroundColor Cyan
Write-Host "🧪 Testing winget installation..." -ForegroundColor Cyan
try {
    $finalTest = winget --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ SUCCESS: Winget is installed and working!" -ForegroundColor Green
        Write-Host "Version: $finalTest" -ForegroundColor Cyan

        Write-Host "" -ForegroundColor Green
        Write-Host "🎯 You can now use winget commands like:" -ForegroundColor Yellow
        Write-Host "   winget install usbipd" -ForegroundColor White
        Write-Host "   winget search python" -ForegroundColor White
        Write-Host "   winget list" -ForegroundColor White
    } else {
        throw "Command failed"
    }
} catch {
    Write-Host "⚠️ Winget installation may need system restart" -ForegroundColor Yellow
    Write-Host "💡 Try running: winget --version (after restart)" -ForegroundColor Cyan
}

Write-Host "" -ForegroundColor Green
Write-Host "🎉 Winget installation process completed!" -ForegroundColor Green
Write-Host "📚 For help: winget --help" -ForegroundColor Cyan

# Keep window open
Read-Host "Press Enter to exit"


