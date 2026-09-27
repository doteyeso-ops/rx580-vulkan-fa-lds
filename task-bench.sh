#!/usr/bin/env bash
# ============================================================================
# TASK COMPLETION TEST: stock (pre-fork) vs fork (post-findings)
#
# WHAT IS BEING MEASURED
#   Wall-clock time from task submission to VERIFIED completion, not tok/s.
#   Completion is gated by task-verify.py (exact match, 3/3 or fail).
#
# WHY WALL CLOCK AND NOT THROUGHPUT
#   deconfound.sh proved pp is not reproducible ACROSS sessions: the identical
#   config with the flag off measured 20.69 pp (clean matrix) and 25.17 pp
#   (deconfound), a +21.7% gap with no code change. That is larger than the
#   effect this study measures, so any tok/s figure compared across sessions is
#   unsound. Same-session back-to-back A/B pairs are the only trustworthy
#   method here, and time-to-verified-pass is a single self-contained number.
#
# WHAT IS HELD CONSTANT (only ggml-vulkan.dll differs)
#   Same commit a25c9865. Same model, same prompt, same ngl, same -t, same
#   context size, same seed. Side order ALTERNATES per pair so that a
#   monotonic thermal drift cannot masquerade as a consistent fork advantage.
#
# ENVIRONMENT ORDERING
#   STOCK first, then FORK, in P1. FORK first in P2. STOCK first in P3.
#   A consistent fork win in alternating order is much harder to explain by
#   drift than a win in one fixed order.
# ============================================================================

# ============================================================================
# PATHS ARE ENV-OVERRIDABLE -- set these to reproduce on your own machine:
#
#   HERE       repo root holding task-ctx-*.txt / task-key-*.txt
#   MODEL      path to the GGUF
#   STOCK_BIN  llama-cli.exe from the STOCK build   (unpatched ggml-vulkan.dll)
#   FORK_BIN   llama-cli.exe from the FORK build    (patched   ggml-vulkan.dll)
#   LOG        where the raw log lands (default task-bench.log)
#
# Only ggml-vulkan.dll must differ between the two builds. Verify with:
#   md5sum "$(dirname $STOCK_BIN)/ggml-vulkan.dll" "$(dirname $FORK_BIN)/ggml-vulkan.dll"
# ============================================================================

set -u
HERE="${HERE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$HERE" || exit 1
export MSYS2_ARG_CONV_EXCL='*'

MODEL="${MODEL:-K:/llm-lowend/Qwen3.6-35B-A3B-UD-IQ3_S.gguf}"
TAG="${TAG:-main}"
NGL="${NGL:-20}"
NTHREADS="${NTHREADS:-8}"
CTX="${CTX:-4096}"
NGEN="${NGEN:-48}"
PAIRS="${PAIRS:-3}"

FORK_BIN="${FORK_BIN:-$HERE/llama.cpp/build-vulkan/bin/llama-cli.exe}"
STOCK_BIN="${STOCK_BIN:-$HERE/stock-test/bin/llama-cli.exe}"
CTXFILE="${CTXFILE:-$HERE/task-ctx-${TAG}.txt}"

LOG="${LOG:-$HERE/task-bench.log}"
: > "$LOG"
say() { echo "$*" | tee -a "$LOG"; }

# --- the one strict serializer, armed for the whole batch -----------------
. ./gpu-lock.sh
gpu_lock_acquire || { say "ABORT: could not acquire GPU lock"; exit 70; }
trap 'gpu_lock_release' EXIT

say "=== TASK BENCH $(date '+%Y-%m-%d %H:%M:%S') ==="
say "config: ngl=$NGL t=$NTHREADS ctx=$CTX ngen=$NGEN tag=$TAG pairs=$PAIRS"

run_side () {
  # $1 = side label, $2 = binary path. Times the run, then grades it.
  local side="$1" bin="$2" t0 t1 rc verdict
  local errf="task-${TAG}-${side}.err"
  local outf="task-out-${TAG}.${side}.txt"

  say "--- ${side}: start $(date '+%H:%M:%S')"
  t0=$(date +%s)
  # PRE-FLIGHT: the context file must actually reach the model. llama-cli
  # treats -f and -p as mutually exclusive -- passing both makes -p win and
  # the archive is silently discarded, which showed up as a 0.2 KB context
  # and a constant 8.2 t/s. Fail loudly instead of benchmarking nothing.
  if ! grep -q 'ARCHIVE OF FIELD OBSERVATIONS -- BEGIN' "$CTXFILE"; then
    say "ABORT: $CTXFILE has no archive block -- regenerating is required"
    rc=99
  else
  "$bin" \
      -m "$MODEL" \
      -ngl "$NGL" --flash-attn on \
      -c "$CTX" -n "$NGEN" -t "$NTHREADS" \
      -f "$CTXFILE" \
      -rea off \
      -st \
      > "$outf" 2> "$errf"
  rc=$?
  fi
  t1=$(date +%s)

  # POST-FLIGHT: the run must have actually LOADED the model and generated.
  # A Vulkan OOM makes llama exit 0 while writing nothing at all, so a
  # zero-length capture plus a "success" rc is indistinguishable from a real
  # run unless it is checked. This is what turned three model-load failures
  # into six "PASS 3/3" retrieval results.
  if ! grep -aqiE 'ErrorOutOfDeviceMemory|failed to load model|exited with code 1' "$errf"; then
    # The OOM can also strike AFTER generation, leaving a complete, correct
    # answer in stdout followed by "decode() failed". That run graded 3/3 while
    # reporting 0.0 t/s. A pass with no measurable work behind it is a
    # poisoned result, not a fast one, so it must fail too.
    if grep -aqiE 'decode\(\) failed|ErrorOutOfDeviceMemory|allocateMemory.*failed' "$outf"; then
      say "ABORT: ${side} decode failed AFTER generating (correct answers but no compute): $(grep -aiE 'decode|allocateMemory' "$outf" | tail -1)"
      rc=96
    elif [ ! -s "$outf" ]; then
      say "ABORT: ${side} produced no output (empty $outf) -- load likely failed"
      say "  last error: $(grep -aiE 'error|failed' "$errf" | tail -1)"
      rc=98
    fi
  else
    say "ABORT: ${side} hit a load/OOM error: $(grep -aiE 'ErrorOutOfDeviceMemory|failed to load model' "$errf" | tail -1)"
    rc=97
  fi

  if [ "$rc" -eq 0 ]; then
    verdict="$(python task-verify.py "$TAG" "$side" 2>&1 | tee -a "$LOG" | grep -o 'VERDICT:.*' || echo 'VERDICT: FAIL grader-error')"
  else
    # A non-zero rc (pre-flight abort, empty capture, load/OOM error) must
    # never be paired with a grader verdict. The run did not happen, so there
    # is nothing to grade -- printing the grader's output here is how a failed
    # load got reported as a retrieval pass.
    verdict="VERDICT: FAIL run-did-not-execute rc=$rc"
    say "$verdict"
  fi

  # Wall clock via integer seconds -- `bc` is not guaranteed present in MSYS,
  # and a missing bc silently yields an empty string in the log.
  local wall_s=$(( t1 - t0 ))

  # NOTE: llama-cli writes ALL timings to STDOUT, not stderr -- smoke.err came
  # back 0 bytes with every number present in the .txt instead. The greps must
  # match THIS build's actual trailer, which is:
  #     [ Prompt: 53.6 t/s | Generation: 3.8 t/s ]
  # The old patterns assumed "load time: N" / "prompt eval time: N ms / N tokens"
  # / "eval time: ... runs" and a bare "(N t/s)" -- none of which appear in a
  # truncated non-interactive run, so every metric logged as "?" for six runs
  # in a row. Falling back to the compact trailer is what finally produced
  # real numbers.
  local load_s pp_tg pp_s tg_tg
  load_s=$(grep -aoiE 'load time: *[0-9.]+' "$outf" | tail -1 | grep -oE '[0-9.]+')
  pp_tg=$(grep -aoiE 'prompt eval time: *[0-9.]+ *ms */ *[0-9]+ tokens' "$outf" | tail -1 | grep -oE '[0-9.]+' | head -1)
  pp_s=$(grep -aoiE '\[ *Prompt: *[0-9.]+ *t/s' "$outf" | tail -1 | grep -oE '[0-9.]+' | tail -1)
  tg_tg=$(grep -aoiE 'eval time: *[0-9.]+ *ms */ *[0-9]+ runs' "$outf" | tail -1 | grep -oE '[0-9.]+' | head -1)

  # Generation t/s lives only in the compact trailer, so read it separately --
  # it is the headline metric and must be logged, not reconstructed.
  local gen_tps
  gen_tps=$(grep -aoiE 'Generation: *[0-9.]+ *t/s' "$outf" | tail -1 | grep -oE '[0-9.]+' | tail -1)

  say "RESULT side=$side rc=$rc wall=${wall_s}s load=${load_s:-?}s pp=${pp_tg:-?}ms pp_tps=${pp_s:-?} gen_tps=${gen_tps:-?} gen=${tg_tg:-?}ms"
  say "  $verdict"
}

for p in $(seq 1 "$PAIRS"); do
  say "=== PAIR $p ==="
  # NULL-CONTROL MODE: FORCE_SIDE=control points BOTH arms at the SAME binary.
  # Whatever spread that produces IS the noise floor -- it is the direct,
  # same-session measurement of the drift that deconfound.sh reported as
  # +21.7% cross-session. Every real A/B delta must be judged against THIS
  # number, not against the older cross-session figure, because the
  # cross-session number is not a valid comparator for a same-session pair.
  if [ "${FORCE_SIDE:-}" = "control" ]; then
    run_side "ctlA" "$STOCK_BIN"
    run_side "ctlB" "$STOCK_BIN"
    continue
  fi
  if [ $((p % 2)) -eq 1 ]; then
    run_side "stock" "$STOCK_BIN"
    run_side "fork"  "$FORK_BIN"
  else
    run_side "fork"  "$FORK_BIN"
    run_side "stock" "$STOCK_BIN"
  fi
done

say "=== TASK BENCH DONE $(date '+%H:%M:%S') ==="
