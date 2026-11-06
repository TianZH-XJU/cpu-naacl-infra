#!/bin/bash
# CPU-CAST Stage-0 environment probe (plan section 2.2).
# Writes artifacts/env/{system_probe,cache_probe,perf_probe}.txt
set -u
cd "$(dirname "$0")/.."
mkdir -p artifacts/env

{
  date -Is
  uname -a
  lscpu
  grep -m1 '^flags' /proc/cpuinfo
  free -h
  df -h .
  ulimit -a
  echo "--- cgroup v2 ---"
  cat /sys/fs/cgroup/cpu.max 2>/dev/null || true
  cat /sys/fs/cgroup/memory.max 2>/dev/null || true
  cat /sys/fs/cgroup/cpu.stat 2>/dev/null || true
  echo "--- affinity ---"
  taskset -pc $$ 2>/dev/null || true
  echo "--- python ---"
  python3 --version
  python3 -c "
import platform, sys
print(platform.platform())
try:
    import onnxruntime as ort
    print('ORT', ort.__version__)
    print('providers', ort.get_available_providers())
except Exception as e:
    print('ORT error', repr(e))
try:
    import torch
    print('Torch', torch.__version__)
    print('CPU capability', torch.backends.cpu.get_cpu_capability())
except Exception as e:
    print('Torch absent or error', repr(e))
"
} | tee artifacts/env/system_probe.txt

echo "=== cache probe ===" | tee artifacts/env/cache_probe.txt
for f in /sys/devices/system/cpu/cpu0/cache/index*/{level,type,size,coherency_line_size}; do
  [ -f "$f" ] && printf '%s: ' "$f" && cat "$f"
done | tee -a artifacts/env/cache_probe.txt

echo "=== perf probe (current process, no system-wide privileges requested) ===" | tee artifacts/env/perf_probe.txt
if command -v perf >/dev/null 2>&1; then
  perf stat -e cycles,instructions,cache-misses,context-switches,page-faults -- sleep 2 2>&1 | tee -a artifacts/env/perf_probe.txt
else
  echo "perf not installed in this sandbox; hardware counters not measured" | tee -a artifacts/env/perf_probe.txt
fi
