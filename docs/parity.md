# Parity Definition: harmonypy-rs

**Reference:** Python `harmonypy` v2.0.2 at `../harmonypy` (tag `v2.0.2`), consisting of
the Python wrapper (`harmonypy/harmony.py`, `harmonypy/lisi.py`) and the C++ backend
(`src/harmony.cpp`, `src/harmony.hpp`, `src/lisi.hpp`, built with Armadillo 14.2.2
header-only + OpenBLAS, libstdc++).

This document is the contract for the entire port. It may not be narrowed or
reinterpreted without explicit user approval.

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
2. **Same RNG stream**: `std::mt19937` seeded with `random_state`; libstdc++
   `std::uniform_real_distribution<float>(0,1)` semantics (`generate_canonical<float,24>`:
   one u32 draw per call, `float(u32)/2^32f`, clamped to `nextafter(1.0, 0.0)`
   — *without consuming another draw* — when the rounded value reaches 1.0);
   libstdc++ `std::shuffle` semantics
   (paired Fisher-Yates branch + `__gen_two_uniform_ints` + Lemire `_S_nd<u64,u32>`
   downscaling). Cross-validated against a compiled C++ probe.
3. **Same k-means initialization semantics**: `arma::kmeans(Y, X, K, keep_existing, 1,
   false)` replicated exactly (Armadillo 14.2.2 `km_iterate`, single-threaded path:
   two-pass paired squared-distance assignment, strict-< tie-breaking, sum/heft update,
   dead-mean resurrection heuristics).
4. **Same internal float32 arithmetic** (`MATTYPE = arma::fmat`): all algorithm state
   in f32; f64 only at the API boundary (input conversion, output exposure) as in the
   C++/Python boundary.
5. **Same discrete behavior**: identical `kmeans_rounds`, identical number of harmony
   iterations, identical error vs. success outcomes, identical numerical-error
   messages (same text, same `%g`-style float formatting).
6. **Golden parity vs the R harmony2 reference fixtures** at the same thresholds the
   Python tests use (per-PC correlation ≥ 0.99 on pbmc_3500; LISI atol = 0.01).
7. **Differential parity vs the installed Python backend** (harmonypy 2.0.2 wheel):
   all differential runs must satisfy the gates in `docs/differential-gates.md`,
   calibrated empirically and committed. Every differential discrepancy is a severe
   blocker: fix it or make it fail loudly.

## Tolerated variations (closed list)

Everything not listed here must be identical.

1. **Bit-level float differences.** The reference uses OpenBLAS (`sgemm`, `snrm2`,
   OpenMP-independent accumulation orders) inside Armadillo; the Rust port uses plain
   f32 loops. Bitwise equality is impossible; results must agree within the committed
   differential gates (see `docs/differential-gates.md`), which are far tighter than
   the Python suite's statistical thresholds.
2. **Logging transport.** Python logs via `logging`; Rust via the `log` crate facade.
   Message *content* is preserved; formatting/timestamps differ (informational only).
3. **`ncores` parameter.** Controls BLAS thread count via environment variables in the
   Python layer; meaningless in a BLAS-free Rust library. Omitted from the Rust API.
4. **Input container types.** Python accepts pandas/DataFrame/dict-likes; the Rust API
   takes concrete slice/`HashMap` inputs with the same semantics (numeric-column
   extraction of `select_dtypes` is the caller's responsibility).
5. **k-means total failure corner case.** If Armadillo's `km_iterate` resurrection
   heuristic fails (`live_gs` empty), the C++ silently produces a degenerate state;
   the Rust port returns a loud error. (Cannot be exercised by the upstream test
   suite; documented per playbook exception rules.)
6. **Armadillo k-means dead-mean "random last resort"** uses Armadillo's internal RNG;
   the Rust port uses its own generator. Only reachable when a cluster dies and no
   live cluster has ≥ 2 cells — not reachable in the test fixtures.
7. **Objective summation order.** The C++ sums the two large objective terms via
   OpenBLAS `sdot`; the port uses Armadillo's dual-accumulator order, which carries
   ≈ 3e-3 relative f32 accumulation error on the 6.9M-element medium-config sums.
   The formula itself is validated in f64 (see `docs/differential-gates.md`); the
   medium `objective_rel` gate is calibrated accordingly.

## Acceptance criteria

- 100% of the ported Python tests pass in Rust (`cargo test`).
- Differential harness (`scripts/differential.py` + `tests/differential.rs`) passes
  with zero unexplained discrepancies against the installed Python backend.
- RNG module matches the compiled C++ probe output exactly (raw u32 stream,
  uniform_real draws, shuffles).
- `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings` clean; CI workflow
  runs fmt, clippy, tests, and the differential harness.
