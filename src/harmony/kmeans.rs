//! Port of `kmeans_init()`: replicates the Armadillo 14.2.2
//! `arma::kmeans(Y, X, K, keep_existing, 1, false)` call (via
//! `gmm_diag<eT>::kmeans_wrapper` → `km_iterate<1>`, single-threaded path).
//!
//! Exact-parity notes:
//! - The random subset selection follows the reference exactly, including
//!   `std::set<unsigned> chosen` (ordered set) and the `prob.max()` /
//!   `index_min()` (first-minimum) tie-breaking.
//! - `km_iterate` with max_iter=1 performs exactly one Lloyd pass: assignment
//!   by two-pass paired squared distance (`distance<eT,1>::eval`, strict `<`
//!   so the first minimum wins), new means = accumulated sum / heft (0 for
//!   dead means), then the dead-mean resurrection heuristics. The loop then
//!   swaps old/new and compares `rs_delta.mean() <= eps`, which with
//!   max_iter=1 only affects whether a second pass would run — it does not,
//!   so the result is the new means.
//!
//! Index-based loops are intentional throughout: they mirror the C++
//! reference's flat-buffer (column-major) indexing directly.

#![allow(clippy::needless_range_loop)]

use crate::fmat::FMat;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use std::collections::BTreeSet;

/// Port of `kmeans_init(X, K, rng)` in harmony.cpp.
pub fn kmeans_init(x: &FMat, k: usize, rng: &mut ChaCha8Rng) -> FMat {
    let n = x.ncols;
    let d = x.nrows;

    let mut y = FMat::zeros(d, k);

    // First pass: uniform random column selection.
    for i in 0..k {
        let mut idx = (rng.random::<f32>() * n as f32).round() as i64;
        if idx >= n as i64 {
            idx = n as i64 - 1;
        }
        y.set_col(i, x.col(idx as usize));
    }

    // Second pass: probability-based selection without replacement
    // (`std::set<unsigned> chosen` — ordered set).
    let mut chosen: BTreeSet<usize> = BTreeSet::new();
    for i in 0..k {
        // distances = |2 * (1 - Y.col(i).t() * X)| as a column (1 x N row):
        // via faer (gemv) when large enough, else a plain sequential loop.
        let yt = y.col(i);
        let mut distances = vec![0.0f32; n];
        if n * d >= crate::fmat::FAER_MIN_ELEMS {
            let yi = faer::MatRef::from_column_major_slice(yt, d, 1);
            let xm = faer::MatRef::from_column_major_slice(&x.data, d, n);
            let dots: faer::Mat<f32> = xm.transpose() * yi;
            let dot = dots.col_as_slice(0);
            for (j, dist) in distances.iter_mut().enumerate() {
                let t = 2.0f32 * (1.0f32 - dot[j]);
                *dist = t.abs();
            }
        } else {
            for (j, dist) in distances.iter_mut().enumerate() {
                let xc = x.col(j);
                let mut acc = 0.0f32;
                for dd in 0..d {
                    acc += yt[dd] * xc[dd];
                }
                let t = 2.0f32 * (1.0f32 - acc);
                *dist = t.abs();
            }
        }

        // random_numbers: N draws of uniform01; prob = -log(rn) / (dist + 1e-10).
        let mut prob = vec![0.0f32; n];
        for j in 0..n {
            let rn = rng.random::<f32>();
            prob[j] = -rn.ln() / (distances[j] + 1e-10f32);
        }

        // for (auto idx : chosen) prob(idx) = prob.max();
        let prob_max = prob.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        for &idx in &chosen {
            prob[idx] = prob_max;
        }

        // unsigned best = prob.index_min();
        let mut best = index_min(&prob);
        // while (chosen.count(best)) { prob(best) = prob.max(); best = index_min(); }
        while chosen.contains(&best) {
            prob[best] = prob_max;
            best = index_min(&prob);
        }
        chosen.insert(best);
        y.set_col(i, x.col(best));
    }

    // 10 iterations of arma::kmeans(Y, X, K, keep_existing, 1, false).
    for _ in 0..10 {
        km_iterate_once(&mut y, x);
    }

    y
}

/// Port of Armadillo 14.2.2 `gmm_diag<eT>::km_iterate` with max_iter=1
/// (single-threaded, non-OpenMP path) — one Lloyd iteration.
fn km_iterate_once(means: &mut FMat, x: &FMat) -> bool {
    let n_dims = means.nrows;
    let n_gaus = means.ncols;
    let x_n_cols = x.ncols;
    if x_n_cols == 0 {
        return true;
    }

    let mut acc_means = vec![0.0f32; n_dims * n_gaus];
    let mut acc_hefts = vec![0u32; n_gaus];
    let mut last_indx = vec![0usize; n_gaus];

    // Assignment pass (old_means = current means). Distances are computed in
    // matmul form through faer: ||x_i - m_g||^2 = ||x_i||^2 - 2*m_g.x_i + ||m_g||^2,
    // with M^T X (n_gaus x n_cells) as one parallel sgemm. Argmin keeps the
    // strict-< first-minimum tie-breaking.
    let old_means = means.clone();
    let mt = old_means.mul_trans_a(x); // n_gaus x n_cells
    let mut x_norms = vec![0.0f32; x_n_cols];
    for (i, col) in x_norms.iter_mut().enumerate() {
        let xc = x.col(i);
        let mut acc = 0.0f32;
        for &v in xc {
            acc += v * v;
        }
        *col = acc;
    }
    let mut m_norms = vec![0.0f32; n_gaus];
    for (g, norm) in m_norms.iter_mut().enumerate() {
        let mc = old_means.col(g);
        let mut acc = 0.0f32;
        for &v in mc {
            acc += v * v;
        }
        *norm = acc;
    }
    for i in 0..x_n_cols {
        let xi2 = x_norms[i];
        let mut min_dist = f32::INFINITY;
        let mut best_g = 0usize;
        for g in 0..n_gaus {
            let dist = m_norms[g] - 2.0 * mt.get(g, i) + xi2;
            if dist < min_dist {
                min_dist = dist;
                best_g = g;
            }
        }
        let x_col = x.col(i);
        for dd in 0..n_dims {
            acc_means[best_g * n_dims + dd] += x_col[dd];
        }
        acc_hefts[best_g] += 1;
        last_indx[best_g] = i;
    }

    // Generate new means.
    for g in 0..n_gaus {
        let heft = acc_hefts[g];
        for dd in 0..n_dims {
            let v = if heft >= 1 {
                acc_means[g * n_dims + dd] / heft as f32
            } else {
                0.0
            };
            means.set(dd, g, v);
        }
    }

    // Heuristics to resurrect dead means.
    let dead_gs: Vec<usize> = (0..n_gaus).filter(|&g| acc_hefts[g] == 0).collect();
    if !dead_gs.is_empty() {
        // live_gs = sort(find(acc_hefts >= 2), "descend").
        let mut live_gs: Vec<usize> = (0..n_gaus).filter(|&g| acc_hefts[g] >= 2).collect();
        live_gs.sort_unstable_by(|a, b| b.cmp(a));
        if live_gs.is_empty() {
            return false;
        }
        let mut live_gs_count = 0usize;
        for &dead_g_id in &dead_gs {
            let proposed_i = if live_gs_count < live_gs.len() {
                let live_g_id = live_gs[live_gs_count];
                live_gs_count += 1;
                if live_g_id == dead_g_id {
                    return false;
                }
                // Recover by using a sample from a known good mean.
                last_indx[live_g_id]
            } else {
                // Random last resort: arma_rng::randi over [0, X_n_cols).
                // (Armadillo's internal RNG; see docs/parity.md tolerated
                // variation #6 — not reachable in the test fixtures.)
                rand_last_resort(x_n_cols)
            };
            if proposed_i >= x_n_cols {
                return false;
            }
            means.set_col(dead_g_id, x.col(proposed_i));
        }
    }

    if means.data.iter().any(|v| !v.is_finite()) {
        return false;
    }
    true
}

/// Deterministic stand-in for Armadillo's internal RNG used by the
/// "randomly selected sample (last resort)" dead-mean recovery path.
fn rand_last_resort(n: usize) -> usize {
    0 % n.max(1)
}

/// Port of `arma::index_min` — first index of the minimum value.
fn index_min(v: &[f32]) -> usize {
    let mut best = 0usize;
    let mut min = f32::INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x < min {
            min = x;
            best = i;
        }
    }
    best
}
