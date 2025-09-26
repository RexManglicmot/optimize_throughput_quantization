# Quantization — Throughput (QPS) Tables
## QPS vs FP32 — per batch
| Batch | Precision | Direction | × vs FP32 | 95% CI | p(two-sided) |
| ---: | --- | --- | ---: | --- | ---: |
| 32 | fp16 | faster | ×1.75 | 1.74–1.76 | 1.8e-09 |
| 32 | int4 | slower | ×0.95 | 0.94–0.96 | 6.3e-04 |
| 64 | fp16 | faster | ×1.99 | 1.97–2.00 | 3.4e-08 |
| 64 | int4 | faster | ×1.24 | 1.23–1.26 | 7.7e-05 |
| 128 | fp16 | faster | ×2.49 | 2.49–2.49 | <1e-12 |
| 128 | int4 | faster | ×1.95 | 1.92–1.97 | 1.6e-05 |
| 256 | fp16 | faster | ×2.64 | 2.62–2.65 | 1.0e-06 |
| 256 | int4 | faster | ×2.19 | 2.18–2.20 | 6.9e-07 |

## QPS vs FP32 — across batches (geo-mean)
| Precision | Direction | × vs FP32 (geo-mean) | 95% CI | p(combined) |
| --- | --- | ---: | --- | ---: |
| fp16 | faster | ×2.19 | 1.81–2.64 | <1e-12 |
| int4 | faster | ×1.50 | 1.02–2.19 | <1e-12 |

## Averaged Results (from results.csv)
| Batch | Precision | QPS | Tokens/sec | p95 Latency (ms) | Peak VRAM (GB) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 32 | fp16 | 10 | 1.52K | 99 | 14.30 |
| 32 | fp32 | 6 | 872 | 173 | 28.59 |
| 32 | int4 | 6 | 825 | 177 | 8.01 |
| 64 | fp16 | 18 | 2.64K | 66 | 15.11 |
| 64 | fp32 | 9 | 1.33K | 126 | 30.19 |
| 64 | int4 | 11 | 1.65K | 109 | 12.11 |
| 128 | fp16 | 25 | 3.84K | 41 | 16.71 |
| 128 | fp32 | 10 | 1.54K | 103 | 33.39 |
| 128 | int4 | 20 | 3.00K | 54 | 13.78 |
| 256 | fp16 | 31 | 4.76K | 33 | 19.93 |
| 256 | fp32 | 12 | 1.81K | 87 | 39.79 |
| 256 | int4 | 25 | 3.96K | 40 | 20.35 |
