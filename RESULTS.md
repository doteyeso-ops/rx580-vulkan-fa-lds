# Two changes in `ggml-vulkan.cpp` measured on a GCN card — RX 580, Windows + AMD proprietary driver

> **⚠️ Read [`FAQ_CORRECTION.md`](FAQ_CORRECTION.md) first.** The first version
> of this file claimed the test machine ran **Mesa RADV** and that the second
> hunk was inert. Both were wrong.

**Hardware:** Radeon RX 580 (GCN, 8 GB) — **AMD proprietary driver 26.5.2**, not
Mesa RADV.
**Model:** Qwen3.6-35B-A3B-UD-IQ3_S.gguf (13,676,723,168 bytes, MoE, 20 layers offloaded)
**Config:** `-ngl 20 -t 8 -c 4096`, serial runs, single model process at a time.

## The patch

Two insertions / two deletions against commit `a25c9865` (`b1530-a25c9865`),
one file — `ggml/src/ggml-vulkan/ggml-vulkan.cpp`:

```diff
@@ -1251 @@ get_fa_tuning_params_scalar()
- if (device->vendor_id == VK_VENDOR_ID_AMD && device->properties.limits.maxComputeSharedMemorySize == 65536) {
+ if (device->vendor_id == VK_VENDOR_ID_AMD && device->properties.limits.maxComputeSharedMemorySize >= 32736) {

@@ -1840 @@ ggml_vk_load_shaders()
- if ((device->architecture == AMD_GCN) && (device->driver_id != vk::DriverId::eAmdProprietary)) {
+ if (device->architecture == AMD_GCN) {
```

### Which hunk actually did the work — unknown, and that is the finding

The first version claimed hunk 2 was inert because "our card runs RADV, so
`driver != proprietary` was already true". That premise was false: the driver
**is** proprietary, so the stock guard was **false** and the patch **enables** the
GCN MMQ warptile tuning. **Both hunks are operative**, and the measurements
cannot attribute the effect to either one.

- **Hunk 1 (LDS threshold) — operative.** The device reports
  `maxComputeSharedMemorySize = 32768`, which fails the old `== 65536` equality
  and satisfies the new threshold.
- **Hunk 2 (driver gating) — operative.** Removing the
  `driver != eAmdProprietary` exclusion changes the branch outcome on this
  machine, which is exactly the population the first version said it did *not*
  affect.

Attributing the effect requires two additional single-hunk builds. That is the
top follow-up and the main weakness of this work.

## Result

| Metric | stock | patched | delta |
|---|---|---|---|
| **Prompt processing** | 52.27 t/s ± 2.33 | **59.25 t/s ± 1.41** | **+13.36%** |
| **Generation** | 3.37 t/s ± 0.72 | 3.60 t/s ± 0.32 | +6.93% *(not significant)* |

n = 6 pairs per side, ABBA-alternating order so thermal drift cannot masquerade
as a consistent win. All 12 arms passed the deterministic retrieval verifier
3/3, with no OOM and no `0.0 t/s` arm.

**Prompt-processing ranges do not overlap at all:** stock max 55.3, patched min
57.9 t/s. Per-pair deltas +13.31, +18.74, +15.98, +7.23, +10.02, +15.70 —
consistent in sign across all six. Paired bootstrap +13.40%, 95% CI
[+10.39, +16.37], P(≤0) = 0.0000.

### Withdrawn: the "real generation regression"

The first version reported a **−5.81%** generation regression and called it
real. With n=6 the sign flips to **+6.93%**, driven by one contaminated arm
(arm 5 stock at 1.90 t/s against a 3.65 median). Median delta is **+0.00%**;
dropping each side's minimum gives **+7.27%**; the paired bootstrap on generation
has 95% CI [−0.45, +21.69], P(≤0) = 0.0522.

**Generation is unchanged at this sample size.** The regression claim is
withdrawn, and no claim is made in either direction.

## Why the prompt number is trustworthy (the part that matters)

An earlier measurement of an identical configuration drifted **+21.7% across
sessions** — larger than the effect claimed here. A double-digit number measured
against that backdrop is meaningless. So the headline is not the +13.36%; it is
this:

> **Null control — the same binary on both arms, 10 arms, 5 pairs.**
> Identical code produced a **1.89% median / 3.73% p95** spurious delta,
> max 5.50%.

The real effect is **~3.6× the same-session noise floor** and the distributions
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

- **Two confounded hunks** — no attribution. Main weakness.
- **n = 6 per side.** Wide CIs; generation is fragile at this size.
- **One workload** (deterministic retrieval, 4096 ctx, 48 gen tokens), one GPU,
  one driver, one OS. The +13.36% is for *that* configuration. It is not a
  general speedup claim, and it is unverified on Linux, RADV, or RDNA.
- **Prompt-heavy by construction** — a 48-token generation phase biases the
  study toward prompt-side effects.
- **Speed ≠ quality.** 3/3 is a pass/fail gate, not a quality comparison.
- **`load`, `pp`, and `gen` millisecond fields logged as `?`** — this build's
  trailer only exposes the t/s summary. Wall-clock per task was 29–34s.

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
6. **One output filename per side silently destroyed data.** Repeated same-side
   arms overwrote each other; the aggregate log survived but the raw evidence
   did not. Now archived per arm.

Each fix is individually regression-tested against the real artifacts. A harness
that reports PASS on empty output will do this again, quietly.

## Bottom line

Two changes to shared-memory-size matching and driver gating yield
**+13.36% prompt-processing throughput** on an RX 580 under the AMD
proprietary driver — roughly 3.6× the measured same-session noise floor, with
generation unchanged. Two device gates, both conservative to the point of
excluding a large class of consumer hardware, both now opened. But this data
cannot say which gate did it, and the driver identification in the first version
of this report was wrong.
