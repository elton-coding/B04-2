# 用本机 Edge 无头模式把态势报告 HTML 打印为 PDF（中文字体由 Windows 提供）
# 用法：powershell -File tools/html2pdf.ps1 -Html output/report/P1A_test_situation_report.html
param([Parameter(Mandatory=$true)][string]$Html, [string]$Pdf = "")
$edge = @("C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", "C:\Program Files\Microsoft\Edge\Application\msedge.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $edge) { throw "未找到 Microsoft Edge" }
$src = (Resolve-Path $Html).Path
if (-not $Pdf) { $Pdf = [IO.Path]::ChangeExtension($src, ".pdf") }
& $edge --headless=new --disable-gpu --no-pdf-header-footer "--print-to-pdf=$Pdf" ("file:///" + $src.Replace([char]92, [char]47)) 2>$null
Start-Sleep -Seconds 3
Get-Item $Pdf | Select-Object FullName, Length
