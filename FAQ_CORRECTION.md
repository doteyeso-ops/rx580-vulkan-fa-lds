# fa-lds-2hunk — Two changes in `ggml-vulkan.cpp` measured on a GCN card (RX 580, Windows + AMD proprietary driver)

**Correction first, because the earlier version of this repo was wrong.** A
previous writeup here claimed the test machine ran **Mesa RADV** and reported
**49152** bytes of shared memory. `vulkaninfo` on that machine says otherwise:

```
deviceName     = Radeon RX 580 Series
driverName     = AMD proprietary driver
driverInfo     = 26.5.2 (AMD proprietary shader compiler)
driverID       = DRIVER_ID_AMD_PROPRIETARY
maxComputeSharedMemorySize = 32768
```

So this is the **proprietary** driver, not RADV, and the shared-memory figure is
**32768**, not 49152. The "open-source driver is locked out" premise that
motivated the original writeup does not hold on this hardware. That framing is
withdrawn.

**And the consequence for the patch itself is the opposite of what was
published.** Upstream gates the FA tuning branch on an exact match:

```diff
- if (vendor_id == AMD && maxComputeSharedMemorySize == 65536) {
+ if (vendor_id == AMD && maxComputeSharedMemorySize >= 32768) {
```

The adjacent hunk upstream reads:

```diff
- if ((device->architecture == AMD_GCN) && (device->driver_id != vk::DriverId::eAmdProprietary)) {
+ if (device->architecture == AMD_GCN) {
```

On a **proprietary** driver the original condition is **false**, so the GCN MMQ
warptile tuning was **skipped**, and this patch **enables** it. The published
claim that this hunk was "inert on this hardware" was wrong.

**This is a two-hunk change, not a one-liner.** The measurements below cannot
attribute the effect to either hunk individually — that would need a build with
only one of the two applied.

## Headline

| Metric | stock | patched | delta |
|---|---|---|---|
| Prompt processing | 52.27 ± 2.33 t/s | **59.25 ± 1.41 t/s** | **+13.36%** |
| Token generation | 3.37 ± 0.72 t/s | 3.60 ± 0.32 t/s | +6.93% *(not significant)* |

n=6 per side, 12/12 arms `rc=0` and 12/12 verified, ABBA-alternating,
`-ngl 20 -t 8 -c 4096 -n 48 --flash-attn on`, Qwen3.6-35B-A3B-UD-IQ3_S.

Paired bootstrap over pairs: prompt **+13.40%**, 95% CI [+10.39, +16.37],
P(≤0) = 0.0000. Ranges **disjoint** (stock max 55.3 < fork min 57.9).

## The generation number does not survive contact with the raw data

v2 (n=4) reported a **−5.81%** generation regression. v3 (n=6) reports **+6.93%**.
They disagree in *sign*. The cause is one arm:

| arm | side | gen t/s | vs median |
|---|---|---|---|
| 5 | stock | **1.90** | −47.9% |

A single stock arm at 1.90 t/s drags the stock mean down and manufactures an
apparent fork "win". Robust statistics disagree with each other because the
underlying data is too small and too contaminated for a mean to be meaningful:

- raw mean delta: **+6.93%**
- median-based delta: **+0.00%**
- drop each side's minimum, then mean: **+7.27%**

**The honest conclusion is that generation is unchanged, and the ±5–7% figures
in either direction are noise, not effect.** The earlier "real, reproducible
−5.81% generation regression" claim is withdrawn. The paired bootstrap on v3
generation is not significant (P(≤0) = 0.0522, CI crosses zero).

The prompt effect is a different matter: it is **+13.36% with disjoint ranges**,
consistent in sign across both sessions (v2 +10.79%, v3 +13.36%), consistent in
sign in **all 6 of 6** per-pair deltas, and roughly 2.9× the measured same-session
noise floor (1.89% median / 3.73% p95 from a 10-arm same-binary null control).

## Why the null control matters

An earlier identical-configuration comparison showed **+21.7%** cross-session
prompt drift — larger than the effect being measured. Cross-session comparison
is not a valid comparator for that reason. Running the **same binary on both
arms** in the same session (10 arms) puts the actual noise floor at **1.89%
median / 3.73% p95**, which is what makes a ~13% same-session effect legible at
all.

The general lesson is driver-agnostic and survives this correction: *when your
effect is smaller than your known drift, the fix is a better design, not more
samples — add a same-session null control.*

## Honest limits

- **n=6 per side.** Effects are consistent in sign but the confidence intervals
  are wide, and the generation metric is demonstrably fragile at this n.
- **Two confounded hunks.** The measured effect belongs to the pair of changes
  together. Which one does the work is unknown.
- **One GPU, one driver, one OS.** Windows 11 + AMD proprietary 26.5.2. Linux
  and RDNA coverage would materially change the conclusions.
- **One workload.** A deterministic 3-question retrieval task, 4096 ctx, 48
  tokens generated.
- **Speed ≠ quality.** Nothing here tests answer quality.
- The stock build is a local checkout, not an official release; the fork is a
  patched build of the same commit `a25c9865` (`b1530-a25c9865`).

## Reproduce

```bash
TAG=smoke PAIRS=6 bash task-bench.sh              # A/B
TAG=smoke PAIRS=5 FORCE_SIDE=control bash task-bench.sh   # null control
```

Paths are environment-overridable — see the header of `task-bench.sh`. Only
`ggml-vulkan.dll` differs between the two builds (see `PROVENANCE.txt`).

## What would strengthen this

1. Split the two hunks and measure each alone. **This is the single highest-value
   next step** and needs only two more builds.
2. More pairs — the current n makes the generation metric unusable.
3. A long-generation workload, where a decode regression would actually show.
4. A second device or driver, ideally RDNA or RADV, to test the generality claim.

## Credits

- **llama.cpp / ggml** (ggml-org) — the tuning code, the warp-tile tables, and
  the FA path being measured. All of the logic here is upstream's; this repo
  only measures it.
- **Mesa / RADV** and the **AMD proprietary driver** — Vulkan drivers. The
  original writeup credited Mesa RADV as the test platform; that was incorrect,
  and Mesa is not involved on this machine.
- **The model** — `Qwen3.6-35B-A3B-UD-IQ3_S.gguf` (Qwen / unsloth community
  quantisation).

MIT licensed for the harness and writeup. Upstream `llama.cpp` is MIT.
