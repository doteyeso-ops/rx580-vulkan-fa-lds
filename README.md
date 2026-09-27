# fa-lds-2hunk — Two changes in `ggml-vulkan.cpp` measured on a GCN card (RX 580, Windows + AMD proprietary driver)

> **⚠️ Read [`FAQ_CORRECTION.md`](FAQ_CORRECTION.md) first.** An earlier version
> of this repo described the test machine as running **Mesa RADV** with
> **49152** bytes of shared memory. That was wrong: `vulkaninfo` reports
> `DRIVER_ID_AMD_PROPRIETARY` and `maxComputeSharedMemorySize = 32768`. The
> original "open-source driver is locked out" premise is withdrawn, and the patch
> turns out to be **two** operative hunks, not one.

## What this is

A measurement of two small changes to `ggml/src/ggml-vulkan/ggml-vulkan.cpp` in
`llama.cpp`, run on a Radeon RX 580 8 GB under Windows 11 with the AMD
proprietary driver (26.5.2).

| Metric | stock | patched | delta |
|---|---|---|---|
| Prompt processing | 52.27 ± 2.33 t/s | **59.25 ± 1.41 t/s** | **+13.36%** |
| Token generation | 3.37 ± 0.72 t/s | 3.60 ± 0.32 t/s | +6.93% *(not significant)* |

n=6 per side · 12/12 arms clean · ABBA-alternating · paired bootstrap
**+13.40%**, 95% CI [+10.39, +16.37], P(≤0) = 0.0000 · ranges **disjoint**
(stock max 55.3 < patched min 57.9) · sign consistent in **6 of 6** pairs.

## The patch

```diff
- if (vendor_id == AMD && maxComputeSharedMemorySize == 65536) {
+ if (vendor_id == AMD && maxComputeSharedMemorySize >= 32768) {

- if ((device->architecture == AMD_GCN) && (device->driver_id != vk::DriverId::eAmdProprietary)) {
+ if (device->architecture == AMD_GCN) {
```

The first relaxes an exact-match capability check into a threshold. The second
**removes a proprietary-driver exclusion**, which on this machine is
load-bearing: stock *skipped* the GCN MMQ warptile tuning on the proprietary
driver, and this patch enables it.

## Quick facts

| | |
|---|---|
| GPU | Radeon RX 580 (GCN, 8 GB) |
| Driver | AMD proprietary 26.5.2 (`DRIVER_ID_AMD_PROPRIETARY`) |
| Model | `Qwen3.6-35B-A3B-UD-IQ3_S.gguf` (MoE, 13.7 GB) |
| Config | `-ngl 20 -t 8 -c 4096 -n 48 --flash-attn on` |
| Base commit | `a25c9865` / build `b1530-a25c9865` |
| Arms | 12 A/B (v3) + 8 A/B (v2) + 10 null-control, all verified |
| Headline | prompt **+13.36%**, generation **unchanged** |

## Honest limits

- **Two confounded hunks.** The effect belongs to both together. Which one does
  the work is unknown — splitting them is the top follow-up.
- **n=6 per side**, and the generation metric is demonstrably fragile at this n.
- **One GPU, one driver, one OS.** Windows 11, AMD proprietary 26.5.2.
- **One workload.** Deterministic 3-question retrieval, 4096 ctx, 48 tokens.
- **Speed ≠ quality.** Nothing here tests answer quality.
- The earlier "real −5.81% generation regression" claim is **withdrawn**: at n=6
  the sign flips, driven by one contaminated arm.

## Method note (the part worth keeping)

An identical-configuration comparison once showed **+21.7% cross-session
drift** — larger than the effect under study. Cross-session data cannot be
repaired by adding samples. The fix was a **same-session null control**: the
*same binary* on both arms, 10 arms, giving a noise floor of **1.89% median /
3.73% p95**. A ~13% same-session effect is ~3.5× that floor, which is what
makes it legible at all.

## Reproduce

```bash
TAG=smoke PAIRS=6 bash task-bench.sh                     # A/B
TAG=smoke PAIRS=5 FORCE_SIDE=control bash task-bench.sh  # null control
```

`task-verify.py` grades output deterministically 3/3; the harness rejects runs
that fail to load, that OOM after generating, or that report `0.0 t/s`.
Paths are environment-overridable — see the header of `task-bench.sh`.

## Layout

```
FAQ_CORRECTION.md   what was wrong, and what changed
RESULTS.md          full numbers, per-arm table, statistics
GITHUB_WRITEUP.md   long-form writeup, use cases, research relevance
patch.diff          the exact diff
PROVENANCE.txt      binary hashes, build IDs, driver identification
task-bench.sh       harness (guards + null-control mode)
task-verify.py      deterministic 3/3 verifier
bench-v2.log        raw A/B log, 8 arms
bench-v3.log        raw A/B log, 12 arms
bench-control.log   raw null-control log, 10 arms
task-out-smoke.*.txt   per-arm raw model output
```

## Credits

- **[`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp)** — the tuning
  code, warp-tile tables, and FA path being measured. All logic is upstream's;
  this repo only measures it.
- **AMD** — the proprietary Vulkan driver used as the test platform. An earlier
  version credited Mesa RADV; that was incorrect and Mesa is not involved here.
- **Qwen / unsloth** — `Qwen3.6-35B-A3B-UD-IQ3_S.gguf` model and quantisation.

MIT licensed for this package. Upstream `llama.cpp` is MIT.
