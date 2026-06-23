#!/usr/bin/env pwsh
# ============================================================
#  FoodSense Conference Paper - Compile Script
#  Run this script from the IEEE_Conference_Paper directory
# ============================================================
Set-Location 'd:\6th Sem\idp\research paper\IEEE_Conference_Paper'

Write-Host "=== Pass 1: pdflatex ===" -ForegroundColor Cyan
pdflatex -interaction=nonstopmode --enable-installer foodsense_conference.tex

Write-Host "`n=== Pass 2: bibtex ===" -ForegroundColor Cyan
bibtex foodsense_conference

Write-Host "`n=== Pass 3: pdflatex ===" -ForegroundColor Cyan
pdflatex -interaction=nonstopmode --enable-installer foodsense_conference.tex

Write-Host "`n=== Pass 4: pdflatex (final) ===" -ForegroundColor Cyan
pdflatex -interaction=nonstopmode --enable-installer foodsense_conference.tex

Write-Host "`n=== Done! ===" -ForegroundColor Green
# Print page count from log
Select-String -Path "foodsense_conference.log" -Pattern "Output written"
