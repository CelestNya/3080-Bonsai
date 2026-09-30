# Download the pinned KVMem runtime (public release, SHA256-verified) into this repo.
# Package: kvmem v0.16.0-rc3-prism.3 (windows x86_64, cuda12.9, rtx20-30-40-50)
# Source:  https://github.com/kvmem/kvmem-llama.cpp/releases/tag/v0.16.0-rc3-prism.3
$ErrorActionPreference = 'Stop'
$tag  = 'v0.16.0-rc3-prism.3'
$pkg  = "kvmem-$tag-windows-x86_64-cuda12.9-rtx20-30-40-50.zip"
$base = "https://github.com/kvmem/kvmem-llama.cpp/releases/download/$tag"
$root = Split-Path $PSScriptRoot -Parent
$tmp  = Join-Path $env:TEMP $pkg

Invoke-WebRequest "$base/$pkg" -OutFile $tmp
Invoke-WebRequest "$base/$pkg.sha256" -OutFile "$tmp.sha256"
$expected = ((Get-Content "$tmp.sha256") -split '\s+')[0].Trim().ToLower()
$actual   = (Get-FileHash $tmp -Algorithm SHA256).Hash.ToLower()
if ($expected -ne $actual) { throw "SHA256 mismatch: expected $expected got $actual" }
Write-Output "sha256 OK: $actual"

Expand-Archive $tmp -DestinationPath $root -Force
Remove-Item $tmp, "$tmp.sha256" -ErrorAction SilentlyContinue
Write-Output "runtime ready: $root\bin\llama-kvmem-server.exe"
