# Draft PR to `ggml-org/llama.cpp` — NOT YET SUBMITTED

> **Status: draft only.** This has not been posted upstream. It is included in
> this repo as a starting point for discussion.
>
> **⚠️ An earlier version of this draft claimed the test machine ran Mesa RADV
> with 49152 bytes of shared memory, and described a "real and reproducible"
> −5.81% generation regression. All of that was wrong.** The machine runs the AMD
> **proprietary** driver and reports **32768** bytes; the generation regression
> did not survive more data and is withdrawn. See `FAQ_CORRECTION.md`.

## Title (proposed)

`ggml-vulkan: relax two conservative device gates that disable tuning on GCN

Two exact-match guards in ggml-vulkan.cpp silently disable tuning paths for whole
classes of hardware. On an RX 580 (GCN, AMD proprietary driver) both evaluate
false. Relaxing them gives +13.36% prompt-processing throughput, with generation
unchanged within noise.`

## Body (proposed)

### Problem

Two guards in `ggml-vulkan.cpp` are written as exact-match or exclusion tests
against queried device properties, so they are false for any device that does not
match precisely:

**1. `get_fa_tuning_params_scalar()` — exact-match on a queried capability**

```cpp
if (device->vendor_id == VK_VENDOR_ID_AMD && device->properties.limits.maxComputeSharedMemorySize == 65536) {
```

This can only ever be true for a device reporting *exactly* 65536 bytes. The RX
580 reports **32768** (`vulkaninfo`, queried on the test host). Stock therefore
never took the tuned branch, which sets subgroup scheduling for a
4-subgroup-per-SIMD occupancy target.

**2. `ggml_vk_load_shaders()` — driver exclusion**

```cpp
if ((device->architecture == AMD_GCN) && (device->driver_id != vk::DriverId::eAmdProprietary)) {
```

On this host `driverID == DRIVER_ID_AMD_PROPRIETARY`, so the guard is **false**
and the GCN MMQ warptile tuning never runs. I could not find documentation
stating that shipping the proprietary driver on GCN is a deliberate opt-out, so
this may be unintentional — but it is also plausibly a deliberate
crash-avoidance workaround, and **I have not investigated why it was added**.
That question should be answered before landing.

### Proposed change

```diff
- if (device->vendor_id == VK_VENDOR_ID_AMD && device->properties.limits.maxComputeSharedMemorySize == 65536) {
+ if (device->vendor_id == VK_VENDOR_ID_AMD && device->properties.limits.maxComputeSharedMemorySize >= 32768) {

- if ((device->architecture == AMD_GCN) && (device->driver_id != vk::DriverId::eAmdProprietary)) {
+ if (device->architecture == AMD_GCN) {
```

### Measurements

RX 580 8 GB, **AMD proprietary driver 26.5.2**, Windows 11.
`Qwen3.6-35B-A3B-UD-IQ3_S.gguf`, `-ngl 20 -t 8 -c 4096 -n 48 --flash-attn on`.
Both arms are the same commit `a25c9865` (build `b1530-a25c9865`); the only
differing file is `ggml-vulkan.dll` (all shared binaries verified
byte-identical). n=6 per side, serial, ABBA-alternating.

| Metric | stock | patched | delta |
|---|---|---|---|
| Prompt processing | 52.27 ± 2.33 t/s | 59.25 ± 1.41 t/s | **+13.36%** |
| Token generation | 3.37 ± 0.72 t/s | 3.60 ± 0.32 t/s | +6.93% *(n.s.)* |

Per-pair prompt deltas: +13.31, +18.74, +15.98, +7.23, +10.02, +15.70 — same
sign in 6/6. Ranges disjoint (stock max 55.3, patched min 57.9). Paired
bootstrap over pairs: +13.40%, 95% CI [+10.39, +16.37].

**Noise floor.** An identical-configuration comparison once drifted +21.7%
*across sessions* — larger than the effect. So I ran the same binary on both
arms (10 arms): spurious deltas 1.89% median, 3.73% p95, 5.50% max. The effect
is ~3.6× that p95 floor, which is what makes it legible at all.

### What I am not claiming

- **I cannot attribute the effect to either hunk.** Both are operative on this
  hardware and the data is from the pair. Splitting them is the obvious next
  step and I have not done it.
- Generation is **unchanged**, not improved. The bootstrap CI is
  [−0.45, +21.69] — it straddles zero. n=6 is too small to resolve a decode
  effect either way.
- One GPU, one driver, one OS, one workload (deterministic retrieval, 4096 ctx,
  48 generated tokens). **No RADV and no Linux data**, which is where most
  readers live. No RDNA, no NVIDIA/Intel.
- Throughput only. No answer-quality measurement.

### Why this is worth a maintainer's attention

Exact-match device gates are a recurring failure mode: they are invisible on the
developer's machine and silently disable tuning everywhere else. The `== 65536`
check is the clearest example I have found, and it is a good candidate for a
regression test on a device reporting a value other than 65536.

### Questions for reviewers

1. Was the `eAmdProprietary` exclusion deliberate? If it was a workaround, what
   was it working around — I would rather not remove it blind.
2. Should the LDS condition be a threshold, or is a device table preferable?
3. Does the effect reproduce on RDNA, and on RADV/Linux?

### Reproducing

```bash
TAG=smoke PAIRS=6 bash task-bench.sh
TAG=smoke PAIRS=5 FORCE_SIDE=control bash task-bench.sh
```

Harness fails closed on non-zero exit, empty output, model-load failure
signatures, Vulkan OOM, and `0.0 t/s` rates. Every arm must score 3/3 on a
deterministic retrieval verifier before its timing is recorded. Full logs and
per-arm raw output are in this repo.

MIT licensed. Upstream `llama.cpp` is MIT.
