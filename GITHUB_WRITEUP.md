# Two changes in `ggml-vulkan.cpp` measured on a GCN card — RX 580, Windows + AMD proprietary driver

> **⚠️ This document was substantially wrong when first published.** It claimed
> the test machine ran **Mesa RADV** and that the second hunk was **inert**.
> `vulkaninfo` on that machine reports `DRIVER_ID_AMD_PROPRIETARY` and
> `maxComputeSharedMemorySize = 32768`. On a proprietary driver the original
> `driver != eAmdProprietary` guard was **false**, so that hunk is
> **load-bearing**, not inert. Both corrections are explained below, and
> [`FAQ_CORRECTION.md`](FAQ_CORRECTION.md) carries the full erratum.

## 1. Summary

Two small changes to `ggml/src/ggml-vulkan/ggml-vulkan.cpp` in `llama.cpp`
produce a **+13.36% prompt-processing throughput** improvement on a Radeon
RX 580 8 GB running the AMD proprietary Vulkan driver under Windows 11.

Token generation is **unchanged** within the noise of this sample. An earlier
version of this document reported a "real, reproducible −5.81% generation
regression"; that claim is **withdrawn** (Section 5).

| Metric | stock | patched | delta |
|---|---|---|---|
| Prompt processing | 52.27 ± 2.33 t/s | **59.25 ± 1.41 t/s** | **+13.36%** |
| Token generation | 3.37 ± 0.72 t/s | 3.60 ± 0.32 t/s | +6.93% *(n.s.)* |

This is a consumer-hardware performance note and a methodology proposal — not
a claim of general speedup. n = 6 paired arms per side, one workload, one GPU,
one driver, one OS.

---

## 2. Why this is worth publishing

- **The bug is a hard-coded equality against a queried capability.** A check for
  `maxComputeSharedMemorySize == 65536` can only ever be true for devices
  reporting *exactly* that value. This device reports **32768**. That is a silent
  performance cliff, and a clean worked example of the `== <constant>`
  device-gate anti-pattern for anyone auditing similar tables.
- **A driver-conditional gate silently disabled a tuning path for a whole
  population.** The stock `driver != eAmdProprietary` guard meant that on *this*
  machine the GCN MMQ warptile tuning never ran at all. Nothing documents that
  as a deliberate opt-out.
- **The methodological result generalises.** The *null control* (Section 4) is
  the most transferable part: a way to rescue an effect smaller than known
  measurement drift. Anyone benchmarking on heterogeneous or consumer hardware
  can reuse it.
- **A negative result that matters.** The first draft's "real generation
  regression" did not survive more data — the sign flipped. Reporting that is
  part of the contribution.

## 3. The change

One file, `ggml/src/ggml-vulkan/ggml-vulkan.cpp`, 2 insertions / 2 deletions,
against upstream `ggml-org/llama.cpp` commit `a25c9865` (build `b1530-a25c9865`).

**Hunk 1 — operative.** In `get_fa_tuning_params_scalar()`:

```diff
- if (device->vendor_id == VK_VENDOR_ID_AMD && device->properties.limits.maxComputeSharedMemorySize == 65536) {
+ if (device->vendor_id == VK_VENDOR_ID_AMD && device->properties.limits.maxComputeSharedMemorySize >= 32768) {
```

The device reports 32768 bytes, which fails the old equality and satisfies the
new threshold. The tuned branch sets subgroup scheduling to hit a
4-subgroup-per-SIMD occupancy target, which plausibly shows up strongly in
prompt processing (many tokens at once) and weakly per-token during decode.

**Hunk 2 — also operative, contrary to the first draft.** In
`ggml_vk_load_shaders()`:

```diff
- if ((device->architecture == AMD_GCN) && (device->driver_id != vk::DriverId::eAmdProprietary)) {
+ if (device->architecture == AMD_GCN) {
```

This removes a **proprietary-driver exclusion**. The original condition was
`GCN && not-proprietary`; the test machine *is* proprietary, so the original was
**false** and the GCN MMQ warptile tuning was **skipped**. The patch makes the
condition true and **enables** that tuning.

The first draft called this hunk "inert", reasoning that RADV satisfies
`driver != proprietary` and therefore the line changed nothing. The premise was
false — the driver is proprietary — so the conclusion inverted. **Reviewers
should treat the earlier "inert" claim as incorrect, in both directions.**

**Consequence: this is a two-hunk change, and the measurements cannot attribute
the effect to either one.** Distinguishing them requires two additional builds
and is the highest-value next step.

## 4. Method

**Setup** — Radeon RX 580 (GCN, 8 GB), AMD proprietary driver 26.5.2, Windows
11. Model `Qwen3.6-35B-A3B-UD-IQ3_S.gguf` (13,676,723,168 bytes, MoE). Config
`-ngl 20 -t 8 -c 4096 --flash-attn on`, 48 generated tokens.

**Provenance — only the file under test differs.** Every shared binary is
byte-identical between arms; the sole delta is `ggml-vulkan.dll`:

| File | MD5 |
|---|---|
| stock `ggml-vulkan.dll` | `9fcd08894bed` |
| patched `ggml-vulkan.dll` | `86e2f1b44d58` |

Both 45,296,128 bytes. `llama-cli.exe` (`53280113ebf8`), `llama-cli-impl.dll`
(`f8652fd913a8`), `ggml.dll`, `ggml-base.dll`, `llama.dll` — all identical
across arms. A third candidate directory (`llama-vulkan-bin/`, md5 `de9090e9…`)
was excluded as a stale debug build.

**Correctness gate.** Every arm is graded by `task-verify.py`, which checks a
deterministic 3-question retrieval task against a fixed answer key; an arm that
does not score 3/3 is not measured. The harness fails closed on: non-zero exit,
missing or empty output, model-load failure signatures, Vulkan OOM, and
`0.0 t/s` rates.

**Pairing.** ABBA-alternating order, serial, one model process at a time, so
thermal drift cannot produce a consistent one-sided win.

**Per-arm archival.** Each arm's raw stdout is written to its own file
(`task-out-smoke.<arm>-<side>.txt`), so every row in the results table traces
to the exact model output that produced it. The first version of this harness
used one fixed filename per side and silently lost all but the last arm.

### 4.1 The null control (the part we care about most)

An identical configuration had previously drifted **+21.7% across sessions** —
*larger than the effect we were trying to measure.* A double-digit number
measured against that backdrop is unsound no matter how many samples you take.

The fix is not more samples; it is a better design. We ran the **same binary on
both arms** (10 arms, 5 pairs). Identical code produced:

| Null-control statistic | Value |
|---|---|
| Median spurious delta | **1.89%** |
| p95 spurious delta | **3.73%** |
| Max spurious delta | **5.50%** |

Therefore:

- Observed prompt delta **+13.36%** ≈ **3.6× the p95 null floor**.
- **No distribution overlap:** stock max 55.3 t/s < patched min 57.9 t/s.
- Paired bootstrap over pairs, P(≤0): **0.0000**, 95% CI [+10.39, +16.37].
- Sign consistent in **6 of 6** pairs.

The +21.7% was a real phenomenon but was *session-bound*, and it is not a valid
comparator for a same-session paired design. Within a session the noise collapses
to under 4%. **Transferable lesson: when your effect is smaller than your known
drift, add a same-session null control rather than discarding the result.**

## 5. Results, and a withdrawn claim

| Metric | stock | patched | delta | vs null floor |
|---|---|---|---|---|
| **Prompt processing** | 52.27 ± 2.33 t/s | **59.25 ± 1.41 t/s** | **+13.36%** | **~3.6× p95** |
| **Generation** | 3.37 ± 0.72 t/s | 3.60 ± 0.32 t/s | +6.93% *(n.s.)* | within floor |

n = 6 per side. Per-pair prompt deltas: **+13.31, +18.74, +15.98, +7.23, +10.02,
+15.70** — consistent in *sign* across all six. Ranges are disjoint.

### The generation claim is withdrawn

The first draft reported a **−5.81%** generation regression and described it as
real, reproducible, and unexplained. It is not.

| session | n/side | gen delta |
|---|---|---|
| v2 | 4 | −5.81% |
| v3 | 6 | **+6.93%** |

The sign flips. The cause is one contaminated arm — arm 5, stock side, at
**1.90 t/s** against a median of 3.65 (−47.9%). Robust estimators disagree with
each other because the sample is too small and too contaminated for a mean to
mean anything:

- raw mean delta: **+6.93%**
- median delta: **+0.00%**
- drop each side's minimum, then mean: **+7.27%**

The paired bootstrap on generation is **not significant**: 95% CI [−0.45, +21.69],
P(≤0) = 0.0522.

**Conclusion: generation is unchanged, and ±5–7% in either direction is noise at
this n.** A genuine decode-side regression would need a long-generation
workload and substantially more pairs to detect. We are not claiming a decode
regression, and we are not claiming there is none.

## 6. Limitations — read before citing

- **Two confounded hunks.** No per-hunk attribution. This is the main weakness.
- **n = 6 per side.** Wide confidence intervals; the generation metric is
  demonstrably fragile at this sample size.
- **One workload, one GPU, one context size.** The +13.36% is for deterministic
  retrieval at 4096 ctx with 48 generated tokens. It is **not** a general
  speedup claim.
- **Prompt-heavy by construction.** A 48-token generation phase is short, which
  biases the study toward prompt-side effects. Long-generation behaviour is
  unmeasured.
- **Speed ≠ quality.** No answer-quality measurement. Both builds scored 3/3 on
  the retrieval task, but that is a pass/fail gate, not a quality comparison.
- **Not tested:** other prompts, model sizes, context lengths; RDNA; NVIDIA or
  Intel; Linux; RADV; the interaction of the two hunks.
- **`load` / `pp` / `gen` millisecond fields log as `?`** — this build's trailer
  exposes only a t/s summary. Wall-clock per task was 29–34 s.
- **The driver identification was wrong in the first draft** and was found only
  by querying the device. Nothing in the benchmark output could have revealed
  it. A repro on Linux with more pairs and a long-generation workload would be
  genuinely welcome.

## 7. Use cases this enables

- **Cheap local inference for prompt-dominant work** — summarisation, RAG
  ingest, document transformation, batch scoring. These see the gain, and since
  decode is unchanged there is no measured price to pay.
- **A concrete upstream PR.** Minimal diff, clear mechanism, real measurements.
  Most valuable if a maintainer can test RDNA *and* a driver that reports
  exactly 65536, which would show whether the effect is specific to the 32768
  case or to the branch being enabled at all.
- **Auditing other device-gate tables** for the same `== <constant>` and
  driver-exclusion anti-patterns.
- **Reusing the null-control design** for any small-delta measurement on noisy
  consumer hardware.

## 8. Harness bugs worth stealing from

The instructive failures were in the measurement harness, not the patch. All are
regression-tested in this repo:

1. **A failed model load can exit 0.** A Vulkan OOM returned `rc=0` with an empty
   capture — indistinguishable from success.
2. **OOM *after* generation is worse than failing outright.** The model emitted
   a complete, correct answer, then died. It graded **3/3 while reporting
   `0.0 t/s`**. A pass with 0.0 behind it is poisoned, not fast.
3. **`exists()` is not "has content."** A 0-byte file graded clean.
4. **Timing regexes written against upstream docs matched nothing here.** This
   build prints `[ Prompt: 53.6 t/s | Generation: 3.8 t/s ]` and *not*
   `load time:` / `prompt eval time: N ms / N tokens`.
5. **`date +%s.%N` inside `$(( ))`** — a float in integer arithmetic killed the
   first run silently, mid-script.
6. **One output filename per side silently destroyed data.** Repeated same-side
   arms overwrote each other; the aggregate log was fine but the raw evidence
   was unrecoverable. Now archived per arm.

A harness that reports PASS on empty output will do this again, quietly.

## 9. Artifacts

- [`FAQ_CORRECTION.md`](FAQ_CORRECTION.md) — **read this first**: the erratum.
- [`RESULTS.md`](RESULTS.md) — full numbers, per-arm table, provenance.
- `bench-v2.log` — raw A/B log (8 arms).
- `bench-v3.log` — raw A/B log (12 arms).
- `bench-control.log` — raw null-control log (10 arms).
- `task-out-smoke.*.txt` — per-arm raw model output.
- `task-bench.sh` — harness: fail-closed guards + null-control mode.
- `task-verify.py` — deterministic 3/3 retrieval verifier.

## Credits

Built on **`ggml-org/llama.cpp`** by the ggml and llama.cpp contributors. The
tuning code being measured is theirs; the RX 580 test platform and all
measurements are ours. The test platform runs the **AMD proprietary driver**;
an earlier version of this document credited Mesa RADV, which was incorrect —
Mesa is not involved on this machine.
