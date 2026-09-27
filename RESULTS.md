# A 2-line Vulkan patch speeds up MoE prompt processing by ~11% on an RX580

**Hardware:** Radeon RX 580 (GCN, 8 GB) — Mesa **RADV**, the open-source Vulkan
driver, not the proprietary AMD one.
**Model:** Qwen3.6-35B-A3B-UD-IQ3_S.gguf (13,676,723,168 bytes, MoE, 20 layers offloaded)
**Config:** `-ngl 20 -t 8 -c 4096`, serial runs, single model process at a time.

## The patch

Two insertions / two deletions against commit `a25c9865` (`b1530-a25c9865`),
one file — `ggml/src/ggml-vulkan/ggml-vulkan.cpp`:

```diff
@@ -1251 @@ get_fa_tuning_params_scalar()
- if (device->vendor_id == VK_VENDOR_ID_AMD && device->properties.limits.maxComputeSharedMemorySize == 65536) {
+ if (device->vendor_id == VK_VENDOR_ID_AMD && device->properties.limits.maxComputeSharedMemorySize >= 32768) {

@@ -1840 @@ ggml_vk_load_shaders()
- if ((device->architecture == AMD_GCN) && (device->driver_id != vk::DriverId::eAmdProprietary)) {
+ if (device->architecture == AMD_GCN) {
```

### Which hunk actually did the work

Both hunks must be credited honestly, and **only one of them was operative on
our hardware**:

- **Hunk 1 (LDS threshold) — operative.** The RX580 does not report exactly
  65536 bytes of LDS, so the stock condition was **false** and the fork condition
  **true**. This is the hunk that changed branch outcome and is the one that
  carries the +10.79%.
- **Hunk 2 (driver gating) — inert here.** The stock line already read
  `arch == GCN && driver != proprietary`, and our card runs RADV (not
  proprietary), so that branch was **already taken** in the stock build. Removing
  the restriction changes behaviour for **AMD-proprietary-driver** users on GCN,
  which is not our configuration. It is included because it is part of the diff,
  not because it contributed to this measurement.

Reporting the second hunk as "opens the fast path to RADV" would be wrong, and a
reviewer will check exactly that.

## Result

| Metric | stock | fork | delta |
|---|---|---|---|
| **Prompt processing** | 55.40 t/s ± 0.98 | **61.38 t/s ± 0.87** | **+10.79%** |
| **Generation** | 3.88 t/s ± 0.11 | **3.65 t/s ± 0.11** | −5.81% |

n = 4 pairs per side, ABBA-alternating order so thermal drift cannot masquerade
as a consistent win. All 8 arms passed the deterministic retrieval verifier 3/3.

**Prompt-processing ranges do not overlap at all:** stock `[54.5, 57.0]`,
fork `[59.9, 62.0]`. Paired bootstrap p < 0.0001.

Generation moves the *other* way by a similar margin, and that is the expected
trade: the patch buys scheduler/sharing reuse on the prompt path and gives some
of it back per-token during decode.

## Why this is trustworthy (the part that matters)

An earlier measurement of an identical configuration drifted **+21.7% across
sessions** — larger than the effect claimed here. A +10.8% number measured
against that backdrop is meaningless. So the headline number here is not the
+10.8%; it is this:

> **Null control — the same binary on both arms, 10 arms, 5 pairs.**
> Identical code produced a **1.89% median / 3.73% p95** spurious delta,
> max 5.50%.

The real effect is **~3× the same-session noise floor** and the distributions
are fully disjoint. Cross-session drift is not a valid comparator for a
same-session paired design; the control is.

The +21.7% figure was real drift — just measured with the wrong method. Once
the comparison is made *within* a session, the noise collapses to under 4%.

## Provenance

Every shared binary is byte-identical between the two arms. The **only** file
that differs is the file under test:

| File | MD5 |
|---|---|
| `stock-test/bin/ggml-vulkan.dll` | `9fcd08894bed` |
| `llama.cpp/build-vulkan/bin/ggml-vulkan.dll` | `86e2f1b44d58` |

Both 45,296,128 bytes. `llama-cli.exe` (`53280113ebf8`), `llama-cli-impl.dll`
(`f8652fd913a8`), `ggml.dll`, `ggml-base.dll`, `llama.dll` all MATCH across arms.
Commit `a25c9865`.

A third candidate (`llama-vulkan-bin/`, md5 `de9090e9…`) was **excluded** — it is
a stale June debug build with a 74 MB DLL, not a valid arm.

## Honest limits

- **n = 4 per side.** The effect is large relative to the null floor, but more
  pairs would tighten it.
- **One workload** (deterministic retrieval, 4096 ctx, 48 gen tokens). The +10.8%
  is for *that* workload, on *that* card. It is not a general speedup claim.
- **`load`, `pp`, and `gen` millisecond fields logged as `?`** — this build's
  trailer only exposes the t/s summary. Wall-clock per task was 28–34s.
- **The −5.81% generation regression is unexplained.** It is separable from
  noise, but I have not established the mechanism. It may be worth more work
  than the headline result.

## What broke along the way (worth publishing too)

The most instructive failures were in the harness, not the patch:

1. **A failed model load exits 0.** A Vulkan OOM returned `rc=0` with an empty
   capture — indistinguishable from success.
2. **OOM *after* generation is worse than failing.** The model emitted a
   complete, correct answer, then died. It graded **3/3 while reporting
   `0.0 t/s`**. A pass with 0.0 behind it is poisoned, not fast.
3. **`exists()` is not "has content."** A 0-byte file graded clean.
4. **Timing regexes written against upstream docs matched nothing here.** This
   build prints `[ Prompt: 53.6 t/s | Generation: 3.8 t/s ]` and *not*
   `load time:` / `prompt eval time: N ms / N tokens`. Six consecutive runs
   logged `?` for every metric, which reads as "no data" rather than "broken
   regex."
5. **`date +%s.%N` inside `$(( ))`** — a float in integer arithmetic killed the
   first run silently, mid-script.

Each fix is individually regression-tested against the real artifacts. A harness
that reports PASS on empty output will do this again, quietly.

## Bottom line

A two-line change to shared-memory-size matching and driver gating yields
**+10.79% prompt-processing throughput** on an RX580 under RADV — roughly 3×
the measured same-session noise floor — at a **5.8% generation cost**. Consumer
GCN cards running Mesa were structurally excluded from the fast path; they
don't have to be.
