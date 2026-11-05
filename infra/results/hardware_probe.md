# CPU-CAST Hardware Probe

- Timestamp (UTC): 2026-11-05T22:16:19.058109+00:00
- CPU: **Intel(R) Xeon(R) 6985P-C CPU @ 2.30GHz**
- Logical CPUs: 5 (physical cores: 5)
- ISA: avx2, avx512f, avx512_vnni, avx_vnni, avx512_bf16, avx512_fp16, fma
- Memory: 5.92 GiB total / 5.51 GiB available
- Disk free: 5465.66 GiB / 5949.61 GiB
- Virtualization: KVM guest (see flags: hypervisor present)

## Matmul thread scaling (1024x1024 fp32)

| Threads | GFLOP/s |
|---|---|
| 1 | 180.56 |
| 2 | 368.19 |
| 4 | 638.63 |
| 5 | 736.83 |

## Memory bandwidth (256 MiB copy)

- 40.01 GiB/s

## Transformer layer thread scaling (batch=32, seq=256, hidden=384)

| Threads | ms/layer |
|---|---|
| 1 | 253.722 |
| 2 | 140.125 |
| 5 | 85.477 |
