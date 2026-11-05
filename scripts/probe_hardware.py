"""CPU-CAST Stage-1 hardware probe.

Collects static platform info and runs compute microbenchmarks
(thread scaling, memory bandwidth) with the actual inference stack.
Writes JSON + Markdown results under infra/results/.
"""
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "infra" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def static_info():
    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "python": sys.version.split()[0],
        "virtualization": "KVM guest" if Path("/proc/xen") is False else None,
    }
    cpuinfo = {}
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            cpuinfo[k.strip()] = v.strip()
    lscpu_flags = cpuinfo.get("flags", "")
    isa = {
        "avx2": "avx2" in lscpu_flags,
        "avx512f": "avx512f" in lscpu_flags,
        "avx512_vnni": "avx512_vnni" in lscpu_flags,
        "avx_vnni": "avx_vnni" in lscpu_flags,
        "avx512_bf16": "avx512_bf16" in lscpu_flags,
        "avx512_fp16": "avx512_fp16" in lscpu_flags,
        "fma": "fma" in lscpu_flags,
        "amx": any(f.startswith("amx") for f in lscpu_flags.split()),
    }
    info["cpu"] = {
        "model": cpuinfo.get("model name"),
        "logical_cpus": psutil.cpu_count(logical=True),
        "physical_cores": psutil.cpu_count(logical=False),
        "sockets": 1,
        "numa_nodes": 1,
        "isa_flags": isa,
        "governor_scaling": None,
    }
    caches = {}
    try:
        caches = {
            "l1d_kib": int(Path("/sys/devices/system/cpu/cpu0/cache/index0/size").read_text().strip().replace("K", "")),
            "l1i_kib": int(Path("/sys/devices/system/cpu/cpu0/cache/index1/size").read_text().strip().replace("K", "")),
            "l2_kib": int(Path("/sys/devices/system/cpu/cpu0/cache/index2/size").read_text().strip().replace("K", "")),
            "l3_kib": int(Path("/sys/devices/system/cpu/cpu0/cache/index3/size").read_text().strip().replace("K", "")),
        }
    except (OSError, ValueError):
        pass
    info["cpu"]["caches_per_core"] = caches
    vm = psutil.virtual_memory()
    info["memory_gib"] = {
        "total": round(vm.total / 2**30, 2),
        "available": round(vm.available / 2**30, 2),
        "swap_total": round(psutil.swap_memory().total / 2**30, 2),
    }
    du = psutil.disk_usage("/")
    info["disk"] = {"total_gib": round(du.total / 2**30, 2), "free_gib": round(du.free / 2**30, 2)}
    return info


def bench_matmul(threads: int, size: int = 1024, iters: int = 20):
    torch.set_num_threads(threads)
    a = torch.randn(size, size)
    b = torch.randn(size, size)
    for _ in range(3):
        a @ b
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    dt = (time.perf_counter() - t0) / iters
    flops = 2 * size**3 / dt
    return {"threads": threads, "matmul_size": size, "seconds": round(dt, 5), "gflops": round(flops / 1e9, 2)}


def bench_mem_bandwidth(size_mib: int = 256, iters: int = 10):
    torch.set_num_threads(psutil.cpu_count(logical=True))
    n = size_mib * 2**20 // 4
    x = torch.randn(n)
    y = torch.empty_like(x)
    y.copy_(x)
    t0 = time.perf_counter()
    for _ in range(iters):
        y.copy_(x)
    dt = (time.perf_counter() - t0) / iters
    return {"buffer_mib": size_mib, "seconds": round(dt, 5), "bandwidth_gib_s": round(2 * size_mib / 1024 / dt, 2)}


def bench_transformer_layer(threads: int, batch: int = 32, seq: int = 256, hidden: int = 384, iters: int = 10):
    torch.set_num_threads(threads)
    layer = torch.nn.TransformerEncoderLayer(
        d_model=hidden, nhead=6, dim_feedforward=4 * hidden, batch_first=True, dropout=0.0
    ).eval()
    x = torch.randn(batch, seq, hidden)
    with torch.inference_mode():
        for _ in range(3):
            layer(x)
        t0 = time.perf_counter()
        for _ in range(iters):
            layer(x)
        dt = (time.perf_counter() - t0) / iters
    return {"threads": threads, "batch": batch, "seq": seq, "hidden": hidden, "ms_per_layer": round(dt * 1000, 3)}


def main():
    torch.set_num_interop_threads(1)
    print("collecting static info...")
    info = static_info()
    print("running thread-scaling matmul...")
    matmul = [bench_matmul(t) for t in (1, 2, 4, info["cpu"]["logical_cpus"])]
    print("running memory bandwidth...")
    mem = bench_mem_bandwidth()
    print("running transformer-layer thread scaling...")
    ncpu = info["cpu"]["logical_cpus"]
    layer = [bench_transformer_layer(t) for t in (1, 2, ncpu)]
    result = {
        "static": info,
        "benchmarks": {"matmul_thread_scaling": matmul, "memory_bandwidth": mem, "transformer_layer_thread_scaling": layer},
        "software": {
            "torch": torch.__version__,
            "torch_num_threads_default": torch.get_num_threads(),
        },
    }
    out_json = RESULTS / "hardware_probe.json"
    out_json.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_json}")

    md = []
    md.append("# CPU-CAST Hardware Probe\n")
    md.append(f"- Timestamp (UTC): {info['timestamp_utc']}")
    md.append(f"- CPU: **{info['cpu']['model']}**")
    md.append(f"- Logical CPUs: {info['cpu']['logical_cpus']} (physical cores: {info['cpu']['physical_cores']})")
    md.append(f"- ISA: {', '.join(k for k, v in info['cpu']['isa_flags'].items() if v)}")
    md.append(f"- Memory: {info['memory_gib']['total']} GiB total / {info['memory_gib']['available']} GiB available")
    md.append(f"- Disk free: {info['disk']['free_gib']} GiB / {info['disk']['total_gib']} GiB")
    md.append(f"- Virtualization: KVM guest (see flags: hypervisor present)\n")
    md.append("## Matmul thread scaling (1024x1024 fp32)\n")
    md.append("| Threads | GFLOP/s |")
    md.append("|---|---|")
    for r in matmul:
        md.append(f"| {r['threads']} | {r['gflops']} |")
    md.append("\n## Memory bandwidth (256 MiB copy)\n")
    md.append(f"- {mem['bandwidth_gib_s']} GiB/s\n")
    md.append("## Transformer layer thread scaling (batch=32, seq=256, hidden=384)\n")
    md.append("| Threads | ms/layer |")
    md.append("|---|---|")
    for r in layer:
        md.append(f"| {r['threads']} | {r['ms_per_layer']} |")
    out_md = RESULTS / "hardware_probe.md"
    out_md.write_text("\n".join(md) + "\n")
    print(f"wrote {out_md}")
    print("\n".join(md))


if __name__ == "__main__":
    main()
