# rx580-vulkan-fa-lds — Consumer GCN + RADV is locked out of llama.cpp's FA LDS fast path

A one-line change to a hard-coded shared-memory check in `ggml-vulkan.cpp`
yields **+10.79% prompt-processing throughput** on a Radeon RX 580 (GCN) running
Mesa RADV, at a **5.8% generation cost**.

Validated against a **same-session null control** (same binary on both arms):
noise floor 1.89% median / 3.73% p95, so the effect is ~2.9× noise with fully
disjoint measurement ranges.

> **Read [`GITHUB_WRITEUP.md`](GITHUB_WRITEUP.md) first** — full method, the
> mechanism, honest limits, and the correction where a second hunk turned out to
> be inert on our hardware.

## Quick facts

| | |
|---|---|
| GPU | Radeon RX 580 (GCN, 8 GB), Mesa **RADV** |
| Model | `Qwen3.6-35B-A3B-UD-IQ3_S.gguf` (MoE) |
| Config | `-ngl 20 -t 8 -c 4096 --flash-attn on` |
| Base commit | `a25c9865` / build `b1530-a25c9865` |
| Arms | 8 A/B + 10 null-control, all 3/3 verified |
| Headline | prompt **+10.79%**, generation **−5.81%** |
| Significance | 2.9× p95 null floor, no range overlap, bootstrap p < 0.0001 |

## The fix

In `get_fa_tuning_params_scalar()`, one condition:

```diff
- ... maxComputeSharedMemorySize == 65536
+ ... maxComputeSharedMemorySize >= 32768
```

The RX580 never reports exactly 65536 bytes of LDS, so it silently took the slow
path. This is a worked example of the `== <constant>` device-gate anti-pattern.

## Reproduce

```bash
# A/B, ABBA-alternating, fail-closed
TAG=smoke PAIRS=4 bash task-bench.sh

# Null control: SAME binary on both arms -> measures the noise floor
TAG=smoke PAIRS=5 FORCE_SIDE=control bash task-bench.sh
```

`task-verify.py` grades output deterministically 3/3; the harness rejects runs
that fail to load, that OOM after generating, or that report `0.0 t/s`.

## Layout

```
GITHUB_WRITEUP.md   the writeup (start here)
RESULTS.md          full numbers + provenance hashes
task-bench.sh       benchmark harness (guards + null-control mode)
task-verify.py      deterministic 3/3 verifier
bench-v2.log        raw A/B log, 8 arms
bench-control.log   raw null-control log, 10 arms
```

## Credits

Built on [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) by the
ggml / llama.cpp contributors, and on **Mesa RADV** by the Mesa project. The
tuning code being corrected is theirs. The RX 580 platform and all measurements
are ours — reproduced and contributed by the community.
