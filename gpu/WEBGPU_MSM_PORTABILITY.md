# Porting the barretenberg WebGPU MSM to Jolt — Portability Assessment

> **Question:** can the WebGPU MSM built in `barretenberg` (aztec-packages) be
> ported to Jolt, targeting **Metal (native)** and **WebGPU (browser)**?
>
> **Short answer: yes — with a clear split.** The *GPU compute* (the WGSL
> shaders) is directly reusable and, via Rust's `wgpu`, gives you **both Metal
> and WebGPU from one shader codebase**. The *host orchestration* (currently
> TypeScript) has to be **reimplemented in Rust**, not copy-pasted. And the
> single most important piece of new work is a **small-/sparse-scalar
> front-end**, because Jolt's MSMs look nothing like the dense, full-width
> workload the barretenberg MSM was tuned for. Details below.
>
> _Branch: `sb/webgpu-msm` (fresh, off updated `main`). Date: 2026-08-06._

---

## 0. Where the source code lives

The WebGPU MSM is **not on `aztec-packages/master`** — it lives on feature
branches (found via `git ls-remote`), authored under the `sb/` and `zw/`
prefixes:

| Branch | What it holds |
|---|---|
| `sb/webgpu-msm-fresh` | Cleaned-up implementation: `msm_v2.ts`, generated shaders, `mask_scalars`, perfetto tracing. Most recent-looking. |
| `sb/msm-webgpu` | Full tree **+ the design docs** (`MSM_DESIGN_ANALYSIS.md`, `README.md`, `WEBGPU_INTEGRATION_STATUS.md`) + dev harness. |
| `sb/integrate-wgpu-msm` | Adds `.claude/plans/msm-webgpu-rewrite.md` and a BrowserStack harness. |
| `zw/msm-webgpu-experiments-backup` | Zac's experiments backup (field-mul micro-benches). |
| `cb/msm-opt/*`, `wip/move-bucket-walk-to-gpu` | Later native/GPU MSM optimization spikes. |

Everything relevant sits under `barretenberg/ts/src/msm_webgpu/` (TS + WGSL)
plus a C++ WASM hook in
`barretenberg/cpp/src/barretenberg/ecc/scalar_multiplication/webgpu_msm_hook.{cpp,hpp}`
and `webgpu_msm_marshalling.hpp`.

**License:** barretenberg is **Apache-2.0**; Jolt is **Apache-2.0 / MIT**. The
WGSL shaders and algorithm are reusable with attribution (add a `NOTICE`).

---

## 1. What the barretenberg WebGPU MSM actually is

It's a **cuZK-style GPU MSM** (descended from the tal-webgpu / atheonxyz ZPrize
2023 submission), specialized to **BN254 G1** only. Pipeline:

1. `decompose_scalars_booth` — Booth/signed-digit windowing of scalars.
2. `bucket_histogram` + `transpose_*` — build the CSR sparse structure (SMVP).
3. `csr_to_v2_*` — bucket accumulation (sparse matrix–vector product).
4. **BPR** (bucket points reduction) — `ba_planner_v2_*`, `ba_reduce_*`.
5. **Batch-affine inversion** (`ba_*`, Montgomery's trick) — turn projective
   results affine in one shared inversion.
6. Horner combination across windows.

Field/curve math lives in the WGSL under `wgsl/{bigint,field,montgomery,cuzk}/`
(28 shaders): Barrett reduction, Montgomery product (Karatsuba/Yuval variants),
`decompress_g1_bn254`, `mask_scalars`, etc.

**Measured performance** (from `MSM_DESIGN_ANALYSIS.md`, BN254):

| n | WebGPU | WASM-MT | Speedup |
|---|---|---|---|
| 2^16 | 75 ms | 77 ms | ~1× |
| 2^20 | 397 ms | 816 ms | **2.06×** |

The win grows with `n` as fixed costs amortize. Dominant GPU stages: **BPR-1**
(flat in `n`, 37 % at 2^16) and **`ba_inverse`** (linear in `n`, 47 % at 2^20).

**Known un-exploited wins the doc itself flags:** GLV endomorphism decomposition
(halves subtask count — Jolt *already has* GLV in `jolt-crypto`), `mixed_safe`
BPR inner loop, hiding `writeBuffer` latency, and — critically — a **sparse
front-end** (see §4).

**Maturity:** per `WEBGPU_INTEGRATION_STATUS.md`, "nothing yet runs an actual
proof — this branch is plumbing." It's a working, benchmarked MSM kernel wired
into browser barretenberg via a WASM import hook, but not a hardened,
shipped-in-a-prover component.

---

## 2. What Jolt needs on the other side

### 2.1 The integration point is clean

The modular crypto crate already exposes a backend-agnostic MSM:

```rust
// crates/jolt-crypto/src/ec/group.rs
pub trait JoltGroup: ... {
    fn msm<F: Field>(bases: &[Self], scalars: &[F]) -> Self;   // <-- the hook
    fn scalar_mul<F: Field>(&self, scalar: &F) -> Self;
    ...
}
```

A GPU MSM implements/overrides `JoltGroup::msm` for `Bn254G1`. It also fits the
`JoltBackend` seam in `jolt-kernels` (the same seam a CUDA/Metal compute backend
would use — see `OPPORTUNITIES.md` §3.1). Call sites that would benefit
immediately: `crates/jolt-dory/src/{scheme,routines,streaming}.rs` and
`crates/jolt-hyperkzg/src/{kzg,scheme}.rs`.

### 2.2 Curve math lines up exactly

Both are **BN254 G1**, `y² = x³ + 3`, ~254-bit base field. Grumpkin (used by
some Jolt inlines) is BN254's cycle partner and is *also* in barretenberg. So
the WGSL field/curve arithmetic is semantically correct for Jolt with **zero
curve changes**.

### 2.3 …but the *scalar workload* is completely different — this is the crux

The barretenberg MSM is tuned for **dense, full-width (254-bit) scalars**. Jolt's
hottest MSMs are the opposite. The legacy MSM entry
(`crates/jolt-prover-legacy/src/msm/mod.rs`) dispatches on scalar width and has
dedicated fast paths for **binary, u8, u16, u32, u64, i128** — because Jolt's
committed polynomials are overwhelmingly **one-hot / small-scalar / very sparse**
(the Twist-&-Shout `Ra` one-hot indices, small increments, etc.).

`MSM_DESIGN_ANALYSIS.md` addresses exactly this and is blunt about it:

- **32-bit scalars** cut subtask count `T` roughly in half → ~40 ms vs 75 ms at
  2^16 (modelled).
- **90 % zero / sparse scalars**: "the current WebGPU pipeline gains almost
  nothing (every kernel still iterates over `n`). A sparse front-end (zero-filter
  compaction prior to decompose) is the single change that would convert this
  distribution into a ~10× win."

**Implication:** porting the code as-is gives you a correct GPU MSM that helps
the *dense* MSMs (e.g. large-scalar Dory commitments), but leaves most of Jolt's
one-hot/sparse MSM volume on the table. The high-value engineering is the
**small-/sparse-scalar front-end** — which does **not exist** in the barretenberg
code and is the main thing worth building on top of the port.

---

## 3. Portability, component by component

| Component | Portable? | How |
|---|---|---|
| **WGSL shaders** (`wgsl/*`, 28 files) | ✅ **Directly** | WGSL is language-agnostic GPU code. Drive them from Rust with [`wgpu`](https://github.com/gfx-rs/wgpu). `wgpu` compiles WGSL to **MSL for Metal**, SPIR-V for Vulkan, and runs natively on WebGPU in-browser — **one shader set, both targets.** |
| **Field representation** | ⚠️ **Marshalling needed** | The shaders use a **13-bit × 20-limb** BN254 field layout (`curve_config.ts: wordSize=13`; the design doc's `w=13, L=20`). Jolt/arkworks uses **64-bit × 4-limb Montgomery**. Convert at the boundary: Jolt `Fr`/`Fq` → little-endian bytes (Montgomery-strip) → GPU limbs, and back (Montgomery-rewrap). Barretenberg already solved this exact pattern in `webgpu_msm_marshalling.hpp` — use it as the spec. |
| **Host orchestration** (`cuzk/*.ts`, `batch_msm.ts`, `msm_v2.ts`, `gpu_context.ts`, `shader_manager.ts`) | 🔁 **Reimplement in Rust** | This is TypeScript driving the WebGPU API (buffer allocation, dispatch scheduling, SAB bridge to WASM). It's the *reference algorithm*, not portable code. Rewrite against `wgpu`'s Rust API. Non-trivial but mechanical; the TS is the executable spec. |
| **C++ WASM hook / SAB bridge** (`webgpu_msm_hook.*`, `bridge/*`) | ❌ **Not needed** | This exists only to call a browser GPU from WASM barretenberg across a SharedArrayBuffer. Jolt is native Rust — it calls `wgpu` directly, so this whole layer disappears (its *marshalling* logic is still a useful reference, per row 2). |
| **Point decompression** (`decompress_g1_bn254`) | ✅ Reusable | Handy if SRS bases are stored compressed. |
| **Batch-affine inversion, BPR, transpose, bucket histogram** | ✅ Reusable | The core cuZK kernels — the valuable part. |

### 3.1 The Metal + WebGPU unification (why this is the right base)

You asked for **Metal and WebGPU**. The reason this port is attractive rather
than two separate efforts:

```
                WGSL shaders (from barretenberg, Apache-2.0)
                              │
                    Rust host via `wgpu`
                    ┌─────────┴───────────┐
              native build            wasm32 build
                    │                      │
             wgpu → Metal            wgpu → WebGPU
             (macOS/iOS GPU)         (browser)
```

One shader codebase, one Rust host, two backends selected at build/runtime.
`wgpu` also gives Vulkan/DX12 for free (Linux/Windows GPUs), which is a bonus for
the broader GPU-backend opportunity.

---

## 4. Verdict & the shape of the work

**Portable: yes.** Roughly:

- **~60 %** of the value (the GPU kernels) transfers with attribution + a `wgpu`
  host rewrite.
- **~30 %** is the Rust `wgpu` orchestration + the field-marshalling boundary —
  well-understood, mechanical, with a reference implementation to follow.
- **~10 % but the highest-leverage** is *new* work: the **small-/sparse-scalar
  front-end** that makes a GPU MSM actually pay off on Jolt's one-hot workload.
  Without it, the port helps only the dense MSMs.

This is a real project, not a weekend port — but the hardest parts (correct,
fast BN254 WGSL kernels; batch-affine inversion; the cuZK pipeline) are already
done and benchmarked.

---

## 5. Proposed integration architecture in Jolt

A new feature-gated crate:

```
crates/jolt-gpu/
├── Cargo.toml            # wgpu; cfg for native (Metal/Vulkan) vs wasm32 (WebGPU)
├── NOTICE                # attribution: WGSL shaders derived from barretenberg (Apache-2.0)
├── src/
│   ├── lib.rs            # GpuMsmContext: device/queue/pipeline setup (once), cached SRS bases
│   ├── marshal.rs        # Bn254 Fr/Fq <-> GPU 13x20 limb LE bytes (Montgomery strip/rewrap)
│   ├── msm.rs            # dense MSM: decompose→SMVP→BPR→batch-affine→horner
│   ├── sparse.rs         # NEW: zero-filter compaction + small-scalar path (Jolt's win)
│   └── shaders/          # vendored WGSL from barretenberg (Apache-2.0)
└── tests/                # equivalence vs jolt-crypto CPU msm() over random + one-hot inputs
```

Wiring: implement `Bn254G1::msm` (or a `GpuMsmBackend` behind the `JoltBackend`
registry) to dispatch to `jolt-gpu` when a `gpu` feature is on and a device is
available, falling back to the existing CPU path otherwise. Validate
**byte-for-byte against `jolt-crypto`'s CPU `msm`** on both random and one-hot
inputs (the same equivalence discipline `jolt-kernels` already uses).

---

## 6. Phased plan

1. **Spike (native/Metal first).** New `crates/jolt-gpu`, pull in the WGSL,
   stand up a `wgpu` device, run the `bigint`/`field`/`montgomery` shaders, and
   validate one BN254 field multiply against `jolt-crypto`. Proves the toolchain
   end-to-end on Metal.
2. **Marshalling.** Implement `marshal.rs` and test round-trips
   `Fr/Fq ↔ 13×20 limbs` against arkworks. (Port `webgpu_msm_marshalling.hpp`.)
3. **Dense MSM correctness.** Wire the full cuZK pipeline in Rust; match
   `jolt-crypto::msm` on random inputs at n = 2^16…2^20.
4. **Integrate at one call site.** Route Dory's large-scalar commitment MSM
   through it behind the `gpu` feature; measure with the profiling harness
   (`crates/jolt-profiling`, perfetto/`summary.json`).
5. **Sparse/small-scalar front-end (the real win).** Add zero-filter compaction
   + a small-scalar (u8/u16/binary) fast path so Jolt's one-hot MSMs actually
   accelerate. This is the piece barretenberg's code lacks.
6. **GLV.** Fold in `jolt-crypto`'s existing BN254 endomorphism decomposition to
   halve subtask count (the design doc's ranked #3 lever; free-ish given Jolt
   already has GLV).
7. **WebGPU target.** Add the `wasm32` build path; verify the same shaders run
   in-browser (COOP/COEP, `@webgpu/types`). Now Metal + WebGPU from one source.
8. **Broaden.** Extend to more MSM call sites; consider a general GPU sumcheck
   backend next (there are already `sb/sumcheck-webgpu` experiments upstream).

---

## 7. Risks & open questions

- **Sparse front-end is unproven for MSM at this specific layout** — highest
  technical risk, but also the highest reward; worth an early feasibility spike
  in parallel with step 3.
- **`ba_inverse` scaling** dominates large-`n` GPU time (47 % at 2^20). The
  multi-workgroup kernel exists but needs `NUM_SUB_WGS` tuning per device (Metal
  vs mobile WebGPU differ).
- **Device portability of WGSL** — the branches carry Adreno-safe variants
  (`read_bits_adreno_safe`) and per-vendor tuning; expect Metal-specific tuning.
- **wgpu maturity for compute + WASM** is good but pipeline-cache and timing
  APIs differ across native/WebGPU; budget for backend-specific glue.
- **CPU↔GPU transfer / dispatch latency** can erase the win at small `n`; keep
  the CPU fallback and a size threshold.
- **Which upstream branch is canonical?** `sb/webgpu-msm-fresh` looks most
  recent; `sb/msm-webgpu` has the docs. Confirm before vendoring shaders.

---

## 8. Immediate next step

Say the word and I'll **vendor the WGSL shaders** from the chosen branch into
`crates/jolt-gpu/src/shaders/` (with a `NOTICE`), scaffold the crate with the
`wgpu` device setup + `marshal.rs`, and stand up the step-1 field-multiply
equivalence test on this branch — so you have a compiling, testable starting
point rather than just this analysis.

---

_Source: `AztecProtocol/aztec-packages` branches `sb/webgpu-msm-fresh`,
`sb/msm-webgpu`, `zw/msm-webgpu-experiments-backup` (barretenberg
`ts/src/msm_webgpu/`, `MSM_DESIGN_ANALYSIS.md`), cross-referenced against this
fork's `crates/jolt-crypto` and `crates/jolt-prover-legacy/src/msm`._
