# Jolt — Optimization & Project Opportunities

> A working document tracking high-value engineering opportunities in and on top of
> [Jolt](https://github.com/a16z/jolt) (the a16z zkVM) and
> [Jolt Atlas](https://github.com/ICME-Lab/jolt-atlas) (NovaNet/ICME Labs' zkML fork).
>
> **Fork status:** this fork's `main` was synced to upstream a16z/jolt `main`
> (`7eaaf1277`, ~PR #1714, Aug 2026), moving it forward ~1,096 commits from the
> December 2024 "Just One Lookup Table" era to the current modular-prover /
> Twist-&-Shout / Dory / Akita architecture.
>
> _Last updated: 2026-08-06._

---

## 1. Where Jolt is right now (the lay of the land)

Understanding the current moment is essential to picking good work, because Jolt is
**mid-migration** and several subsystems are deliberately half-built.

### 1.1 Jolt zkVM (a16z/jolt)

- **RISC-V target:** now **RV64IMAC** (was RV32I in the old fork).
- **Default proving stack:** BN254 scalar field + **Dory** PCS + Blake2b transcript +
  **Twist/Shout** memory-checking + Spartan R1CS, all sumcheck-driven.
- **Two provers coexist:**
  - `crates/jolt-prover-legacy` — the mature monolith. **Still the production path**:
    `jolt-sdk` and the root CLI link *only* legacy; `#[jolt::provable]` uses it.
  - `crates/jolt-prover` — the new **modular, backend-agnostic prover** (orchestration
    only) sitting on ~26 supporting crates. Its `optimized` kernel tier **just hit
    legacy-parity perf** (PR #1714) and gained **BlindFold ZK** (PR #1690). Not yet
    wired into the SDK.
- **The kernel seam:** `crates/jolt-kernels` puts *all* heavy field compute behind a
  swappable `JoltBackend` registry, with a `reference/` (naive, O(K·T), auto-derived,
  semantic ground truth) tier and an `optimized/` (hand-tuned, byte-for-byte
  equivalence-tested) tier.
- **PCS options:** Dory (default), HyperKZG (`crates/jolt-hyperkzg`, secondary),
  **Akita lattice PCS** (`crates/jolt-akita`, newly integrated, post-quantum/transparent,
  actively perf-tuned).
- **ZK:** BlindFold (`crates/jolt-blindfold`) — Pedersen-committed sumcheck rounds whose
  verifier checks are encoded as R1CS and Nova-folded (no SNARK recursion).
- **Official roadmap** (`book/src/roadmap/`): **streaming prover** (in progress),
  **recursion** (planned, docs stub), **on-chain verifier** (planned, docs stub — the
  old Solidity `jolt-evm-verifier` is gone).

### 1.2 Jolt Atlas (ICME-Lab/jolt-atlas)

- A **zkML framework** that adapts Jolt's lookup-centric proving to **ONNX tensor
  computation** instead of RISC-V execution. Thesis: prove non-linear ML ops (ReLU,
  softmax, tanh, erf) as **lookup arguments over sumcheck** rather than arithmetic
  circuits. Headline: nanoGPT ~2.3 s, GPT-2 (125M) ~17 s on an M3/16 GB.
- **Crates:** `atlas-onnx-tracer` (ONNX front-end via `tract-onnx`, quantizes to
  fixed-point i32, executes to produce a trace), `jolt-atlas-core` (per-node sumcheck
  prover/verifier), `joltworks` (a **vendored hard fork** of Jolt's crypto core,
  ML-specialized), `common`.
- **Key divergence from the zkVM:** the running prover has **no Twist / offline
  memory-checking / Spartan-R1CS** in its main path. Data-flow between tensor ops is
  enforced by an **opening-accumulator + eval-reduction** chaining scheme over a
  reverse-topological DAG walk. (The whitepaper's §2 "staged DAG" describes the older
  CPU-trace model; §5.4 documents the tensor-trace reality that the code actually
  implements.)
- **Defaults to HyperKZG** (per-tensor commitments are small; eyes on-chain verification),
  with Dory selectable.

---

## 2. Prioritized shortlist

Ranked by (impact × tractability) for someone building *on* Jolt who wants to add real
value. Details for each in §3–§4.

| # | Opportunity | Where | Impact | Effort | Type |
|---|-------------|-------|--------|--------|------|
| 1 | **GPU/accelerator backend** for `jolt-kernels` | `crates/jolt-kernels` | 🔥🔥🔥 | High | Greenfield |
| 2 | **Close jolt-atlas ZK soundness gaps** | `jolt-atlas-core/.../zk.rs` | 🔥🔥🔥 | Med | Finish-the-feature |
| 3 | **Finish modular-prover migration** (SDK wiring + kernel gap audit) | `jolt-prover`, `jolt-sdk` | 🔥🔥 | Med | Migration |
| 4 | **Akita lattice PCS to perf target** (≤0.5× Dory) | `crates/jolt-akita` | 🔥🔥 | Med-High | Perf |
| 5 | **zkML precompiles: bridge jolt-atlas ↔ Jolt inlines** | new crate | 🔥🔥 | High | Novel project |
| 6 | **Streaming prover** completion | `subprotocols/streaming_*` | 🔥🔥 | High | Roadmap |
| 7 | **Non-power-of-two shapes** in jolt-atlas | `jolt-atlas-core`, tracer | 🔥 | Med | Correctness |
| 8 | **On-chain (EVM) verifier** for the modular proof | new crate | 🔥🔥 | High | Roadmap |
| 9 | **Softmax aux-vector elimination** + fused exp | `ops/softmax_last_axis` | 🔥 | Med | Perf |
| 10 | **Reproducible benchmark/telemetry harness** across both repos | new tooling | 🔥 | Low-Med | Enabler |

---

## 3. Core Jolt zkVM opportunities

### 3.1 🔥 GPU / accelerator backend for `jolt-kernels`  *(flagship)*

**What.** There is currently **no GPU backend** in Jolt. `jolt-kernels` was explicitly
designed so compute implementations "swap without touching protocol structure" — the
`JoltBackend` registry is the exact hook a CUDA/Metal backend would plug into. GPU
Montgomery-form constants already exist (`crates/jolt-field/src/arkworks/montgomery_impl.rs`,
"constants from the original bn254_fr.metal shader"), i.e. someone already started the
field-arithmetic groundwork.

**Why it matters.** GPU acceleration is on a16z's stated roadmap but unbuilt. The
sumcheck inner loop (poly `bind`, `sumcheck_evals`, `eq` evals) and MSM/commitment
dominate prover time and are embarrassingly parallel — the canonical zk GPU win. This is
the single biggest greenfield lever and is *architecturally invited*.

**Where to start.** Target one hot kernel end-to-end first: the Spartan outer sumcheck or
the RAM read-write kernel in `crates/jolt-kernels/src/optimized/`. Implement a
`CudaBackend`/`MetalBackend` behind `JoltBackend`, validate byte-for-byte against the
`reference/` tier (the equivalence-test harness already exists), then expand kernel
coverage. Field ops → MSM → sumcheck binding, in that order.

**Effort:** High. **Impact:** 🔥🔥🔥 (multi-× prover speedup potential).

### 3.2 🔥 Finish the modular-prover migration

**What.** `jolt-prover` reached legacy parity (#1714) but is **not production-wired** —
`jolt-sdk/Cargo.toml` still depends only on `jolt-prover-legacy`. Two concrete tracks:

1. **Wire the modular prover into the SDK / `#[jolt::provable]`** behind a feature flag and
   drive the cutover.
2. **Audit optimized-kernel gaps.** Any relation lacking an `optimized/` kernel silently
   falls back to the O(K·T) `reference` path. Some `optimized/` kernels are still thin
   wrappers that construct `NaiveSumcheckProver` (e.g.
   `crates/jolt-kernels/src/optimized/ram_raf_evaluation.rs`). Grep for
   `NaiveSumcheckProver::new` under `optimized/` and replace each with a real kernel.

**Why.** Unblocks everything downstream (GPU backend, Akita, ZK) that targets the modular
architecture rather than the legacy monolith.

**Effort:** Med. **Impact:** 🔥🔥.

### 3.3 🔥 Bring the Akita lattice PCS to its perf target

**What.** Akita (post-quantum, transparent lattice PCS) is integrated (#1655/#1675/#1676)
but below target: `specs/akita-perf-plan.md` targets **≤0.5× Dory** prove time; last
recorded state is ~0.60× at 2^20, with a flagged upstream regression (akita #302's
quantum-SIS cutover made one-hot setup ~400× slower — the prepared-setup cache is called
"urgent"). Named levers: witness ownership/copy removal, column-derivation reuse, grouped
one-hot commitment (vs sparse-union), Hachi-style layout, prepared-setup cache.

**Why.** A competitive PQ/transparent PCS removes Jolt's trusted-setup dependence and is
strategically important; each lever is a self-contained, measurable win.

**Where:** `crates/jolt-akita/src/scheme.rs`, `specs/akita-perf-plan.md`.

**Effort:** Med-High. **Impact:** 🔥🔥.

### 3.4 🔥 Streaming prover

**What.** Goal (`book/src/roadmap/streaming.md`): bounded prover memory (~2 GB / 1M cycles,
<2× overhead) for arbitrarily long executions **without SNARK recursion** — the streaming
alternative to continuations. Only the **Spartan outer sumcheck** streams today
(`crates/jolt-prover-legacy/src/subprotocols/streaming_sumcheck.rs`,
`streaming_schedule.rs`, wired in `zkvm/spartan/outer.rs`). The rest of the pipeline is
not yet stream-ified.

**Why.** This is *the* differentiator vs other zkVMs and the enabler for proving on
mobile/edge. References NTZ25 (eprint 2025/611) and BCFFMMZ25 (eprint 2025/1473) —
multi-round-per-pass streaming sumcheck.

**Where to start.** Extend the streaming schedule to registers and instruction-lookup
sumchecks next; Twist/Shout's first 128 rounds are already streaming-amenable.

**Effort:** High. **Impact:** 🔥🔥.

### 3.5 On-chain (EVM) verifier for the modular proof

**What.** The old Solidity `jolt-evm-verifier` is gone and `book/src/roadmap/on-chain-verifier.md`
is a stub. A modern on-chain verifier for the current Dory/HyperKZG proof format is
missing. A KZG-family PCS (HyperKZG, `crates/jolt-hyperkzg`) is the natural on-chain path
(pairing-based succinct openings); Dory's GT operations are expensive on-chain.

**Why.** On-chain verification is a headline use case (rollups, bridges) and currently has
no path. High strategic value.

**Effort:** High. **Impact:** 🔥🔥.

### 3.6 Recursion / proof aggregation

**What.** `book/src/roadmap/recursion.md` is a stub; the `examples/recursion` example only
runs the verifier *inside a guest* (recursive verification), not true proof
aggregation/composition. Building a proper aggregation layer (fold/aggregate N proofs into
one) is open.

**Effort:** High. **Impact:** 🔥.

### 3.7 Micro-perf & small-value optimizations (good first PRs)

Concrete, self-contained TODOs flagged in legacy code (each a bounded win):

- `zkvm/spartan/instruction_input.rs:219` — "3-round compression SVO on each of the 8
  multilinears, then bind directly" (unimplemented small-value optimization).
- `zkvm/instruction_lookups/read_raf_checking.rs:924` — use unreduced arithmetic in
  `eval_linear_prod_assign`.
- `zkvm/spartan/product.rs:70` — unify the product sumcheck with Spartan outer/inner
  (structural dedup).
- `poly/commitment/dory/jolt_dory_routines.rs:134` — "this can be optimized heavily?"
- `utils/expanding_table.rs:59` — "this is bad parallelisation."
- `subprotocols/mles_product_sum.rs:779` — "Implement more optimal way to do this."

**Effort:** Low-Med each. **Impact:** 🔥 (individually), good on-ramp to the codebase.

### 3.8 Tracer completeness & configurability

The RV64IMAC emulator (`tracer/src/emulator/`) carries the highest TODO density in the repo
(~42): memory protection & SV48 paging unimplemented (`mmu.rs`), several "@TODO: Optimize"
(`mmu.rs`, `cpu.rs`), a hardcoded trace-buffer capacity that should be configurable
(`cpu.rs:388`), unsupported C.LQ (128-bit), and a large block of missing CSR/trap/privilege
test coverage (`cpu.rs:1462-1493`). Relevant to soundness/coverage rather than prover perf.

**Effort:** Med. **Impact:** 🔥 (correctness/coverage).

### 3.9 Docs debt (low effort, real value)

The entire `book/src/how/optimizations/` chapter (small-value, EQ, compression,
batched-sumcheck/openings) plus the recursion & on-chain roadmap pages are 🚧 stubs.
Writing these from the now-landed code is high-leverage community value and forces a deep
read of the optimizations.

---

## 4. Jolt Atlas (zkML) opportunities

### 4.1 🔥 Close the ZK soundness gaps  *(highest leverage in atlas)*

`jolt-atlas-core/src/onnx_proof/zk.rs` has ~17 `unimplemented!`/`panic!` sites and, more
seriously, real soundness holes:

- **zk.rs:1926** — the BlindFold R1CS **does not yet enforce the sumcheck output** (a
  soundness gap in the ZK path).
- **zk.rs:1515** — a **disabled range-check** ("Re-enable when range check constraints are
  debugged").
- Unimplemented ZK flows for clamp-based `SoftmaxLastAxis`, all clamped activations
  (tanh/erf/sigmoid), scalar fused-rescale, and `MeanOfSquares` — meaning ZK effectively
  covers only a fraction of real models today.

**Why.** ZK is the product's headline claim; the gaps mean it doesn't hold end-to-end for
transformer models. Clearest "finish the feature (correctly)" opportunity.

**Effort:** Med. **Impact:** 🔥🔥🔥.

### 4.2 🔥 zkML precompiles: bridge jolt-atlas ↔ Jolt inlines  *(novel project)*

**What.** Jolt's zkVM has a rich **inlines/precompile** system
(`jolt-inlines/{sha2,keccak256,blake2,blake3,bigint,secp256k1,grumpkin,p256}`) that proves
crypto primitives constraint-natively instead of as guest CPU cycles. Jolt Atlas proves ML
ops via bespoke sumchecks. There is no bridge. A project that exposes **ML ops (matmul,
attention, softmax) as Jolt zkVM precompiles**, or conversely lets an atlas proof invoke
zkVM inlines for embedded crypto (e.g. verifying a signature over a model's output inside
one proof), would let a single Jolt proof cover *program + ML inference* — a genuinely new
capability (verifiable agents, on-chain ML with crypto attestation).

**Why.** Neither repo does this; it's where "projects on top of Jolt that add value" is
most differentiated.

**Effort:** High. **Impact:** 🔥🔥.

### 4.3 Non-power-of-two shape support

~15 tests are `#[ignore = "TODO: non-power-of-two ... path not fully validated yet"]`
across einsum, gather, sum-axis, broadcast, moveaxis, iff, and trig ops. Everything pads to
powers of two; the non-pow2 correctness story is shaky and padding wastes prover work.
Un-ignoring and fixing these removes foot-guns and cuts waste.

**Effort:** Med. **Impact:** 🔥 (correctness + some perf).

### 4.4 Softmax auxiliary-vector elimination + fused exp

Softmax ships `max_k`, `exp_sum_q`, `argmax_k` **in the clear** — O(F) field elements added
to every proof (`ops/softmax_last_axis/mod.rs`, TODO #218) — and its exponentiation still
uses a non-fused multiply (`exponentiation/mult.rs` "refactor to use mul sum-check").
Deriving these in-protocol and fusing the multiply cuts proof size and transcript cost on
the single most-used transformer op.

**Effort:** Med. **Impact:** 🔥.

### 4.5 Streaming / small-space proving for large einsums

The whitepaper advertises prefix-suffix streaming (O(|T|^{1/C}) memory) but the prover
materializes witness polys per node. Implementing streaming for the large einsum "whales"
(attention/MLP/logits) realizes the memory-scaling win the paper claims but the code
doesn't yet deliver — directly analogous to §3.4 and could share technique.

**Effort:** High. **Impact:** 🔥🔥.

### 4.6 Broaden ONNX op coverage

Finish `MeanOfSquares` (LayerNorm building block; still has a fallback arm and is
unimplemented in ZK), general **multi-axis** reductions, `Gather` with `axis≠0`,
`Scan`/control-flow (currently panics), per-node/per-layer **scales** (a single global
compile-time `MODEL_SCALE=8` is baked in and asserted at runtime), and automatic rebase-node
insertion (currently manual). Each unlocks more model families.

**Effort:** Med (per op). **Impact:** 🔥.

### 4.7 Reduce fork divergence from upstream Jolt

`joltworks` is a **hard fork** of Jolt's crypto core, so it silently misses upstream
Twist/Shout and prover improvements. Two options: (a) a periodic re-sync discipline, or
(b) refactor `joltworks` into a **thin adapter over the real `jolt-*` crates** so atlas
tracks upstream automatically. (b) is more work but structurally durable.

**Effort:** Med-High. **Impact:** 🔥 (compounding, risk-reduction).

### 4.8 Post-quantum / on-chain PCS for atlas

The `CommitmentScheme` trait makes PCS a drop-in point (HyperKZG/Dory/mock are already
interchangeable). Swapping in a lattice multilinear PCS (Akita/Hachi/ABBA) addresses both
PQ security and on-chain-verification cost — and could reuse §3.3's Akita work across both
repos.

**Effort:** Med-High. **Impact:** 🔥.

---

## 5. Cross-cutting enablers

- **Reproducible benchmark & telemetry harness** across *both* repos: standardize on the
  perfetto/`summary.json` taxonomy (`crates/jolt-profiling/src/taxonomy.rs`) and
  `jolt-eval` objectives, and publish a stable set of workloads (fibonacci/sha-chain for
  the zkVM; nanoGPT/GPT-2/BGE for atlas). Every perf item above needs this to be credible;
  low effort, unblocks everything. **Do this first if pursuing any perf work.**
- **Expand formal verification.** `z3-verifier` and the Rust→Lean bytecode extractor
  (`jolt-lean-gen`, #1666) exist but coverage is partial. Extending Z3/Lean coverage to
  more relations raises assurance on soundness-critical code.
- **Fill the missing correctness tests** flagged in code (e.g. `compute_all_G` and its
  sumcheck in `zkvm/claim_reductions/hamming_weight.rs:821-822`).

---

## 6. Suggested first moves

If picking one thing to start now, in order of recommendation:

1. **Stand up the benchmark harness (§5)** — a day or two, and it makes every subsequent
   perf claim measurable.
2. **GPU backend spike (§3.1)** — one hot kernel behind `JoltBackend`, validated against
   the reference tier. Highest ceiling, and the architecture is waiting for it.
3. **Close atlas ZK gaps (§4.1)** — if the interest is zkML, this is the highest-integrity
   contribution and directly fixes advertised-but-broken behavior.
4. **A micro-perf PR (§3.7)** — lowest-risk way to learn the modular codebase before
   committing to a large project.

---

_Sources: direct reading of a16z/jolt `main` @ `7eaaf1277` (code, `book/`, `specs/`,
`CLAUDE.md`, git history) and ICME-Lab/jolt-atlas `main` (code, `papers/`), plus a16z
crypto's public posts on the [6× speedup](https://a16zcrypto.com/posts/article/jolt-6x-speedup/)
and [Twist and Shout](https://a16zcrypto.com/posts/article/introducing-twist-and-shout/),
and the [Jolt Atlas whitepaper](https://arxiv.org/abs/2602.17452)._
