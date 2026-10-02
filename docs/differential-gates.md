# Differential gates (self-golden)

The fixtures under `tests/fixtures/differential_*.json` are **self-goldens**:
they are generated from the Rust implementation itself (via the ignored
`tests/gen_golden.rs` generator), because the algorithm is fully deterministic
for a given `random_state` (a `ChaCha8Rng` seeded from it). They are a
regression lock-in rather than a cross-check against Python.

## Two kinds of gates

1. **`check_exact` — lock-in.** The current run must reproduce the committed
   fixture within a tight tolerance. This catches any unintentional change to
   the algorithm, matrix ops, or seeding. Regenerate the fixtures with
   `cargo test --release --test gen_golden -- --ignored` only when a change is
   intentional.

   | config          | data                          | z_corr | y     | objectives |
   |-----------------|-------------------------------|--------|-------|------------|
   | small_defaults  | pbmc3k (3500 cells), defaults | 5e-4   | 5e-4  | 1e-3 rel   |
   | iter2           | pbmc3k, max_iter_harmony=2    | 5e-4   | 5e-4  | 1e-3 rel   |
   | fixed_lamb      | pbmc3k, fixed lamb            | 5e-4   | 5e-4  | 1e-3 rel   |
   | multi_cov       | synthetic 400x5, 2 covariates | 5e-4   | 5e-4  | 1e-3 rel   |
   | medium          | ircolitis (68785, d=50)       | 5e-4   | 5e-4  | 5e-3 rel   |

   `kmeans_rounds` must match exactly in every config.

2. **`check_quality` — statistical sanity**, independent of any particular
   oracle:
   - all outputs (`z_corr`, `y`, `r`, objectives) are finite (no NaN/Inf);
   - k-means makes progress in every round (`kmeans_rounds >= 1`);
   - per-PC variance is broadly preserved by the correction (ratio within
     [0.2, 5.0]);
   - batch mixing (mean LISI across cells, perplexity 30) does not degrade by
     more than 10% after correction (skipped for the 69k medium config to keep
     the kd-tree LISI cost down).

## Why tolerance instead of bit-exact

The actual run-to-run drift is at the f64-ulp level (≈1e-15) because faer's
multithreaded GEMM reduces in a scheduling-dependent order. The stored values
are f32-derived anyway, so a tight absolute bound (≈5e-4) sits far above that
noise while remaining far below any real algorithmic change. `kmeans_rounds`
is discrete and is still required to match exactly.

## Historical note

These fixtures originally came from the installed Python harmonypy 2.0.2
backend (which seeds k-means from mt19937), and the gates compared point-wise
against it. That cross-check was dropped when the RNG was replaced with
`ChaCha8Rng`: a different RNG converges to an equivalent-but-not-identical
solution, making point-wise parity against a specific (RNG-dependent) Python
output unattainable. The statistical-quality checks are what now demonstrate
the result is a good, reproducible Harmony correction.
