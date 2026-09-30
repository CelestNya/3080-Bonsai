@echo off
rem ============================================================
rem Bonsai-2-27B (abliterated+MTP) 64K full-speed mode (no KVMem)
rem For 10GB VRAM cards. 12GB cards: raise nothing, KV is tiny here.
rem Single-shot: no watchdog. Port 29187. Log: service.log
rem ============================================================
cd /d %~dp0..
set KVDIR=%~dp0..\bin
set MODEL=%~dp0..\models\bonsai-abliterated-mtp.gguf
netstat -ano | findstr ":29187" | findstr "LISTENING" >nul
if not errorlevel 1 (
  echo [%DATE% %TIME%] port 29187 busy, not starting >> service.log
  echo port 29187 already in use - not starting
  exit /b 1
)
nvidia-smi -pl 320 >nul 2>&1
set GGML_CUDA_BATCH_INVARIANT=1
set GGML_CUDA_PTQ1_0_MMQ_MAX_BATCH=0
echo [%DATE% %TIME%] starting 64k >> service.log
"%KVDIR%\llama-kvmem-server.exe" -m "%MODEL%" --jinja -n 4096 -ngl 99 -t 8 -fa on --parallel 1 -c 131072 --kvmem-budget 40960 --kvmem-gen-reserve 8192 --kvmem-cpu-gb 2 --cache-type-k q8_0 --cache-type-v q4_0 --host 127.0.0.1 --port 29187 >> service.log 2>&1
echo [%DATE% %TIME%] EXITED code=%ERRORLEVEL% >> service.log
