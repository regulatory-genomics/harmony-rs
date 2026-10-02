# Parity Definition: harmonypy-rs

**Reference:** Python `harmonypy` v2.0.2 at `../harmonypy` (tag `v2.0.2`), consisting of
the Python wrapper (`harmonypy/harmony.py`, `harmonypy/lisi.py`) and the C++ backend
(`src/harmony.cpp`, `src/harmony.hpp`, `src/lisi.hpp`, built with Armadillo 14.2.2
header-only + OpenBLAS, libstdc++).

This document is the contract for the entire port. It may not be narrowed or
reinterpreted without explicit user approval.

> **Amendment (2026-10-02, user-approved).** The original contract demanded
> bit-exact replication of the compiled C++ backend. Per user approval, this was
> relaxed to **golden-gate parity**: all five differential golden tests must
> pass (see `docs/differential-gates.md`); bit-level equality inside k-means and
> the linear-algebra kernels is no longer required. This enables the `faer`
> integration below.
>
> **Amendment 2 (2026-10-02, user-approved).** The RNG was replaced with Rust's
> own `ChaCha8Rng` (seeded from `random_state`) — see "Same RNG" below —
> so the RNG stream is no longer `std::mt19937`-compatible. The differential
> fixtures were reframed as **self-goldens** (generated from the Rust port
> itself) with **statistical-quality** gates replacing point-wise parity against
> Python's mt19937-derived output. The requirements and acceptance criteria
> below are amended accordingly.

## Scope

`harmonypy-rs` is a pure-Rust library crate that is a drop-in replacement for the
harmonypy Python package **at the library API level**:

- `run_harmony(...)` — all parameters, same defaults, same argument normalization,
  same shape/orientation handling, same error conditions.
- `compute_lisi(...)` — same inputs, same output.
- Result object exposing `Z_corr`, `Z_orig`, `Z_cos`, `R`, `Y`, `K`,
  `objective_harmony`, `objective_kmeans`, `kmeans_rounds` with the same shapes as
  the Python `Harmony` object (N×d for Z_corr/Z_orig/Z_cos/R, d×K for Y).

## Parity requirements (mandatory)

1. **Same algorithm, step-by-step**, following `src/harmony.cpp` (which itself follows
   the R harmony2 package): identical iteration structure, block updates, ridge
   correction paths (single-covariate closed-form inverse, multi-covariate
   `arma::inv`-equivalent), convergence checks, and state validation.
2. **Same RNG semantics**: a `ChaCha8Rng` (via `rand_chacha`) seeded from
   `random_state` (`seed_from_u64(random_state as u64)`), so results are fully
   deterministic and reproducible for a given seed. k-means initialization and
   the block-update shuffle draw from it. This replaces the former
   `std::mt19937` + libstdc++ emulation; the exact mt19937 *stream* is no longer
   replicated (see Amendment 2).
3. **Same k-means initialization semantics**: `arma::kmeans(Y, X, K, keep_existing, 1,
   false)` replicated (Armadillo 14.2.2 `km_iterate`: strict-< tie-breaking,
   sum/heft update, dead-mean resurrection heuristics). Distances are computed in
   matmul form (`||x_i||² − 2·m_g·x_i + ||m_g||²` through `faer`, with a plain-loop
   fallback below the size threshold) rather than the C++'s paired per-cell loop;
   the assignment/reselection *decisions* are structurally identical.
4. **Same internal float32 arithmetic** (`MATTYPE = arma::fmat`): all algorithm state
   in f32; f64 only at the API boundary (input conversion, output exposure) as in the
   C++/Python boundary. Large matrix products run through `faer`'s f32 kernels
   (SIMD + rayon-parallel), matching the reference's OpenBLAS-backed sgemm in role;
   products below `FAER_MIN_FLOPS`/`FAER_MIN_ELEMS` (`src/fmat.rs`) use plain f32
   loops, where faer's parallel dispatch would cost more than the arithmetic.
5. **Same discrete behavior**: identical `kmeans_rounds`, identical number of harmony
   iterations, identical error vs. success outcomes, identical numerical-error
   messages (same text, same `%g`-style float formatting).
6. **Golden parity vs the R harmony2 reference fixtures** at the same thresholds the
   Python tests use (per-PC correlation ≥ 0.99 on pbmc_3500; LISI atol = 0.01).
7. **Self-golden + statistical-quality gates** (`tests/differential.rs`): the
   committed fixtures are generated from the Rust port itself (deterministic for a
   given seed) and the run must reproduce them within the committed tolerances
   (see `docs/differential-gates.md`), plus the statistical-quality checks
   (finite outputs, variance preservation, k-means progress, LISI batch mixing
   not degrading).

## Tolerated variations (closed list)

Everything not listed here must be identical.

1. **Bit-level float differences.** The reference uses OpenBLAS (`sgemm`, `snrm2`,
   OpenMP-independent accumulation orders) inside Armadillo; the Rust port uses
   `faer` (different blocked summation orders) and plain f32 loops. Bitwise equality
   is impossible; results must agree within the committed differential gates
   (see `docs/differential-gates.md`). Because faer's multithreaded GEMM reduces
   in a scheduling-dependent order, consecutive runs can differ by an f64 ulp, so
   the self-golden lock-in uses a tight absolute bound (~5e-4) rather than exact
   equality; `kmeans_rounds` is discrete and still matches exactly.
2. **RNG-driven divergence from the Python reference.** Replacing `mt19937` with
   `ChaCha8Rng` changes the (random) k-means initialization and block-update
   shuffle, so the corrected embedding is an equivalent-quality but quantitatively
   different solution than Python's mt19937-based output. This is why the fixtures
   are now self-goldens validated by statistical-quality checks rather than
   point-wise comparison against Python (see Amendment 2).
3. **Logging transport.** Python logs via `logging`; Rust via the `log` crate facade.
   Message *content* is preserved; formatting/timestamps differ (informational only).
4. **`ncores` parameter.** Controls BLAS thread count via environment variables in the
   Python layer; in the Rust library it is superseded by `faer`'s rayon-backed
   parallelism (the global rayon pool). Omitted from the Rust API.
5. **Input container types.** Python accepts pandas/DataFrame/dict-likes; the Rust API
   takes concrete slice/`HashMap` inputs with the same semantics (numeric-column
   extraction of `select_dtypes` is the caller's responsibility).
6. **k-means total failure corner case.** If Armadillo's `km_iterate` resurrection
   heuristic fails (`live_gs` empty), the C++ silently produces a degenerate state;
   the Rust port returns a loud error. (Cannot be exercised by the upstream test
   suite; documented per playbook exception rules.)
7. **Armadillo k-means dead-mean "random last resort"** uses Armadillo's internal RNG;
   the Rust port uses its own generator. Only reachable when a cluster dies and no
   live cluster has ≥ 2 cells — not reachable in the test fixtures.
8. **Objective summation order.** The C++ sums the two large objective terms via
   OpenBLAS `sdot`; the port uses Armadillo's dual-accumulator order, which carries
   ≈ 3e-3 relative f32 accumulation error on the 6.9M-element medium-config sums.
   The formula itself is validated in f64 (see `docs/differential-gates.md`); the
   medium `objective_rel` gate is calibrated accordingly.

## Acceptance criteria

- 100% of the ported Python tests pass in Rust (`cargo test`).
- Self-golden + statistical-quality harness (`tests/differential.rs`) passes with no
  tolerated-gate violations; `kmeans_rounds` matches the committed fixtures exactly.
- RNG is deterministic for a given `random_state`: same seed → same result (modulo
  faer's ulp-level GEMM scheduling noise), different seeds → different results.
- `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings` clean; CI workflow
  runs fmt, clippy, tests, and the differential harness.
- Performance: `tests/bench.rs` (run with `cargo test --release --test bench --
  --ignored --nocapture`) times `run_harmony` on the pbmc3k (small) and ircolitis
  (medium) configs as the committed speed baseline.
