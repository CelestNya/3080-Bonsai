# Service + hardware status
try { $p = Invoke-RestMethod -Uri 'http://127.0.0.1:29187/props' -TimeoutSec 8; Write-Output ("serving: " + $p.model_name) } catch { Write-Output "service: DOWN" }
nvidia-smi --query-gpu=memory.used,memory.total,power.draw,clocks.sm --format=csv,noheader
$os = Get-CimInstance Win32_OperatingSystem
Write-Output ("RAM free GB: " + [math]::Round($os.FreePhysicalMemory/1MB,2))
Get-Content (Join-Path $PSScriptRoot '..\service.log') -Tail 3 -ErrorAction SilentlyContinue
