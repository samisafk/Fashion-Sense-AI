# Fashion Sense AI - Flask Startup Script
# Run this script to start the Flask application

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "Fashion Sense AI - Flask API Server" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (-not (Test-Path ".\venv\Scripts\Activate.ps1")) {
    Write-Host "Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run: python -m venv venv" -ForegroundColor Yellow
    exit 1
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1

# Set protobuf environment variable
Write-Host "Setting environment variables..." -ForegroundColor Yellow
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = 'python'

# Check if Flask dependencies are installed
Write-Host "Checking dependencies..." -ForegroundColor Yellow
$flaskInstalled = pip list | Select-String "Flask"
if (-not $flaskInstalled) {
    Write-Host "Installing Flask dependencies..." -ForegroundColor Yellow
    pip install -r requirements_flask.txt
}

Write-Host ""
Write-Host "Starting Flask application..." -ForegroundColor Green
Write-Host "Access the app at: http://localhost:5000" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host ""

# Run Flask app
python flask_app.py
