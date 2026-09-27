# Consumer GCN + Mesa RADV is locked out of llama.cpp's flash-attention LDS fast path — a 1-line fix, +10.8% prompt throughput

**TL;DR** — Lowering one hard-coded shared-memory check in `ggml-vulkan.cpp`
from `== 65536` to `>= 32768` yields **+10.79% prompt-processing throughput** on
a Radeon RX 580 (GCN) running Mesa **RADV**, at a **5.8% generation cost**. The
effect is **~2.9× a same-session null-control noise floor (3.73%)** with fully
disjoint measurement ranges.

This is a consumer-hardware performance note and a methodology proposal — not a
claim of general speedup. n = 4 paired arms per side, one workload, one GPU.
Section 6 states the limits plainly.

---

## 1. Why this is worth publishing

- **Large installed base.** RX 580 / GCN is a high-volume used-market card, and
  Mesa RADV is the *default* Vulkan driver on Linux. Consumer AMD cards running
  open-source drivers are a large, under-served group in llama.cpp tuning, and
  for many people the consumer card is the *only* GPU available for local
  inference.
- **The bug is a hard-coded equality against a vendor constant.** A check for
  `maxComputeSharedMemorySize == 65536` is a silent performance cliff: it can
  only ever be true for devices reporting *exactly* that value. Any other device
  silently takes the slow path. This is a general anti-pattern in hand-tuned
  device tables, and it makes a good worked example for anyone auditing similar
  code.
- **The methodological result generalizes.** Arguably the most transferable part
  is the *null control* (Section 4): a way to rescue an effect smaller than
  known measurement drift. Anyone benchmarking on heterogeneous or consumer
  hardware can reuse it.

## 2. The change

One file, `ggml/src/ggml-vulkan/ggml-vulkan.cpp`, 2 insertions / 2 deletions,
against upstream `ggml-org/llama.cpp` commit `a25c9865` (build `b1530-a25c9865`).

**Hunk 1 — operative.** In `get_fa_tuning_params_scalar()`:

```diff
- if (device->vendor_id == VK_VENDOR_ID_AMD && device->properties.limits.maxComputeSharedMemorySize == 65536) {
+ if (device->vendor_id == VK_VENDOR_ID_AMD && device->properties.limits.maxComputeSharedMemorySize >= 32768) {
```

The RX580 does not report exactly 65536 bytes of LDS, so **stock never took the
tuned branch**. The tuned branch sets subgroup scheduling to hit a 4-subgroup-
per-SIMD occupancy target — which plausibly shows up strongly in prompt
processing (many tokens at once) and weakly per-token during decode.

**Hunk 2 — inert on our hardware.** In `ggml_vk_load_shaders()`:

```diff
- if ((device->architecture == AMD_GCN) && (device->driver_id != vk::DriverId::eAmdProprietary)) {
+ if (device->architecture == AMD_GCN) {
```

The stock line *already* admitted RADV (`driver != proprietary` is true for
RADV), so this line changed **no** behaviour in our configuration. It affects
**AMD-proprietary-driver** users on GCN, which is not what we measured. We list
it because it is in the diff, and we explicitly **do not** credit it with the
speedup. An earlier draft credited it wrongly; that was a misreading of the
`&&` and is corrected here. Reviewers should treat any claim that this hunk
"enables RADV" as incorrect.

## 3. Method

**Setup** — Radeon RX 580 (GCN, 8 GB), Mesa RADV. Model
`Qwen3.6-35B-A3B-UD-IQ3_S.gguf` (13,676,723,168 bytes, MoE). Config
`-ngl 20 -t 8 -c 4096 --flash-attn on`, 48 generated tokens. llama.cpp
`a25c9865` / `b1530-a25c9865`, Windows 11.

**Provenance — only the file under test differs.** Every shared binary is
byte-identical between arms; the sole delta is `ggml-vulkan.dll`:

| File | MD5 |
|---|---|
| stock `ggml-vulkan.dll` | `9fcd08894bed` |
| fork `ggml-vulkan.dll` | `86e2f1b44d58` |

Both 45,296,128 bytes. `llama-cli.exe` (`53280113ebf8`), `llama-cli-impl.dll`
(`f8652fd913a8`), `ggml.dll`, `ggml-base.dll`, `llama.dll` — all identical across
arms. A third candidate directory (`llama-vulkan-bin/`, md5 `de9090e9…`) was
excluded as a stale debug build.

**Correctness gate.** All 18 arms (8 A/B + 10 control) passed a deterministic
retrieval verifier **3/3** on exact-match injected facts. Runs that failed to
load, or that generated correctly and *then* died, are rejected by the harness
rather than counted (Section 7).

**Pairing.** ABBA-alternating order, serial, one model process at a time, so
thermal drift cannot produce a consistent one-sided win.

## 4. The null control (the part we care about most)

An identical configuration had previously drifted **+21.7% across sessions** —
*larger than the effect we were trying to measure.* A +10.8% number measured
against that backdrop is unsound no matter how many samples you take.

The fix is not more samples; it is a better design. We ran the **same binary on
both arms** (10 arms, 5 pairs). Identical code produced:

| Null-control statistic | Value |
|---|---|
| Median spurious delta | **1.89%** |
| p95 spurious delta | **3.73%** |
| Max spurious delta | **5.50%** |

Therefore:

- Observed real delta **+10.79%** ≈ **2.9× the p95 null floor**.
- **No distribution overlap:** stock `[54.5, 57.0]` t/s, fork `[59.9, 62.0]` t/s.
- Paired bootstrap, P(fork ≥ stock): **p < 0.0001**.

The +21.7% was a real phenomenon but was *session-bound*, and it is not a valid
comparator for a same-session paired design. Within a session the noise collapses
to under 4%. **Transferable lesson: when your effect is smaller than your known
drift, add a same-session null control rather than discarding the result.**

## 5. Results

| Metric | stock | fork | delta | vs null floor |
|---|---|---|---|---|
| **Prompt processing** | 55.40 t/s ± 0.98 | **61.38 t/s ± 0.87** | **+10.79%** | **~2.9× p95** |
| **Generation** | 3.88 t/s ± 0.11 | 3.65 t/s ± 0.11 | −5.81% | above floor |

n = 4 per side. Per-pair prompt deltas: **+8.12, +12.61, +8.77, +13.76** —
consistent in *sign* across all four.

Generation moves the other way by a similar margin. That is the expected trade:
the patch buys prompt-side occupancy and gives some of it back per-token during
decode. The regression is separable from noise but **unexplained** — we have not
established the mechanism and do not claim one.

## 6. Limitations — read before citing

- **n = 4 per side.** Large relative to the null floor, but this is a small
  study; more pairs would tighten it.
- **One workload, one GPU, one context size.** The +10.8% is for deterministic
  retrieval at 4096 ctx. It is **not** a general speedup claim.
- **Generation regression unexplained** — −5.8%, real, mechanism unknown.
- **Not tested:** other prompts, model sizes, context lengths; RDNA; NVIDIA or
  Intel; the AMD proprietary driver. Whether the hunk helps or hurts
  long-generation workloads is **unknown** — and the observed decode cost makes
  that a live question, not a rhetorical one.
- **`load` / `pp` / `gen` millisecond fields log as `?`** — this build's trailer
  exposes only a t/s summary. Wall-clock per task was 28–34 s.
- **Windows 11.** RADV on Windows is far less common than on Linux. The
  mechanism (the LDS size the driver reports) is the same, but we did not verify
  on Linux, where most readers live.

We would rather overstate the limits than understate them. A repro on Linux with
more pairs and a long-generation workload would be genuinely welcome.

## 7. Use cases this enables

- **Cheap local inference for prompt-dominant work** — summarization, RAG
  re-ingest, document transformation, batch scoring. These see the gain; a 5.8%
  decode cost is a fair price.
- **A concrete upstream PR.** Minimal, well-measured, clear mechanism. Most
  valuable if a maintainer can confirm behaviour on RDNA, where the branch was
  presumably intended to apply.
- **Auditing other device-gate tables** for the same `== <constant>` anti-pattern.
  The RX580 is a useful test case precisely because it is *excluded by
  construction* rather than merely slow.

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
   `load time:` / `prompt eval time: N ms / N tokens`. Six consecutive runs
   logged `?` for every metric, which reads as "no data" rather than "broken
   regex."
5. **`date +%s.%N` inside `$(( ))`** — a float in integer arithmetic killed the
   first run silently, mid-script.

A harness that reports PASS on empty output will do this again, quietly.

## 9. Artifacts

- [`RESULTS.md`](RESULTS.md) — full numbers, provenance hashes, limits.
- `bench-v2.log` — raw A/B log (8 arms).
- `bench-control.log` — raw null-control log (10 arms).
- `task-bench.sh` — harness: fail-closed guards + null-control mode.
- `task-verify.py` — deterministic 3/3 retrieval verifier.

## Credits

Built on **`ggml-org/llama.cpp`** by the ggml and llama.cpp contributors, and on
**Mesa RADV** by the Mesa team. The tuning code being corrected is theirs; the
RX 580 test platform and measurements are ours. Thanks to the Hermes agent
harness for the fail-closed benchmark plumbing.
