//! Debug: compare Rust kmeans_init stage-by-stage against the compiled C++
//! reference probes (cpp_stage{1,2,3}_y.bin from kmeans_stages.cpp).
//!
//! Run with: cargo test --test debug_kmeans -- --ignored --nocapture

#![allow(clippy::needless_range_loop)]
mod common;

use harmony::fmat::FMat;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::BTreeSet;

fn data_path(name: &str) -> String {
    format!("{}/data/{}", env!("CARGO_MANIFEST_DIR"), name)
}

fn compare_stage(label: &str, y: &FMat, path: &str) {
    let bytes = std::fs::read(path).unwrap();
    let cpp: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(cpp.len(), y.data.len());
    let mut maxd = 0.0f64;
    let mut n_diff = 0usize;
    let mut worst = 0usize;
    for i in 0..cpp.len() {
        let d = (y.data[i] as f64 - cpp[i] as f64).abs();
        if d > 0.0 {
            n_diff += 1;
        }
        if d > maxd {
            maxd = d;
            worst = i;
        }
    }
    println!(
        "{label}: max abs diff = {maxd:.3e} at {worst} ({n_diff} of {} differ)",
        cpp.len()
    );
}

#[test]
#[ignore]
fn compare_kmeans_stages_medium() {
    let (_, _meta_cols) = common::read_tsv(&data_path("ircolitis_blood_cd8_obs.tsv.gz"));
    let (pcs_header, pcs_cols) = common::read_tsv(&data_path("ircolitis_blood_cd8_pcs.tsv.gz"));
    let (data_mat, n, d) = common::cols_to_matrix(&pcs_cols, &pcs_header);
    let k = 100usize;

    // The row-major N x d buffer is exactly the column-major d x N buffer of
    // X^T (same as the C++ probe's arma::fmat(X32.data(), d, N)), so build the
    // f32 matrix directly.
    let mut out = vec![0.0f32; n * d];
    for (i, &v) in data_mat.iter().enumerate() {
        out[i] = v as f32;
    }
    let z_norm = FMat::from_col_major(d, n, out).normalise_cols2();

    let mut rng = ChaCha8Rng::seed_from_u64(0);

    // ---- Stage 1 ----
    let mut y1 = FMat::zeros(d, k);
    let mut idxs = vec![];
    for i in 0..k {
        let mut idx = (rng.random::<f32>() * n as f32).round() as i64;
        if idx >= n as i64 {
            idx = n as i64 - 1;
        }
        if i < 10 {
            idxs.push(idx);
        }
        y1.set_col(i, z_norm.col(idx as usize));
    }
    println!("rust indices: {:?}", idxs);
    compare_stage("stage1", &y1, "/tmp/opencode/cpp_stage1_y.bin");

    // ---- Stage 2 (exact harmony.cpp code, sequential dot like the port) ----
    let mut y2 = y1.clone();
    let mut chosen: BTreeSet<usize> = BTreeSet::new();
    let mut chosen_order: Vec<usize> = vec![];
    for i in 0..k {
        let yt = y2.col(i);
        let mut distances = vec![0.0f32; n];
        for j in 0..n {
            let xc = z_norm.col(j);
            let mut acc = 0.0f32;
            for dd in 0..d {
                acc += yt[dd] * xc[dd];
            }
            let t = 2.0f32 * (1.0f32 - acc);
            distances[j] = t.abs();
        }
        if i == 0 {
            let bytes = std::fs::read("/tmp/opencode/cppseq_dist0.bin").unwrap();
            let cpp: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            assert_eq!(cpp.len(), n);
            let mut maxd = 0.0f64;
            let mut n_diff = 0usize;
            for i in 0..n {
                let d = (distances[i] as f64 - cpp[i] as f64).abs();
                if d > 0.0 {
                    n_diff += 1;
                }
                maxd = maxd.max(d);
            }
            println!("dist0: max abs diff = {maxd:.3e} ({n_diff} of {n} differ)");
        }
        if i == 95 {
            // Compare distances at the first diverging pick.
            let bytes = std::fs::read("/tmp/opencode/cppseq_dist95.bin").unwrap();
            let cpp: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let mut maxd = 0.0f64;
            let mut n_diff = 0usize;
            for i in 0..n {
                let d = (distances[i] as f64 - cpp[i] as f64).abs();
                if d > 0.0 {
                    n_diff += 1;
                }
                maxd = maxd.max(d);
            }
            println!("dist95: max abs diff = {maxd:.3e} ({n_diff} of {n} differ)");
        }
        let mut prob = vec![0.0f32; n];
        for j in 0..n {
            let rn = rng.random::<f32>();
            prob[j] = -rn.ln() / (distances[j] + 1e-10f32);
        }
        if i == 95 {
            let bytes = std::fs::read("/tmp/opencode/cppseq_prob95.bin").unwrap();
            let cpp: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let mut maxd = 0.0f64;
            let mut n_diff = 0usize;
            for i in 0..n {
                let d = (prob[i] as f64 - cpp[i] as f64).abs();
                if d > 0.0 {
                    n_diff += 1;
                }
                maxd = maxd.max(d);
            }
            println!("prob95: max abs diff = {maxd:.3e} ({n_diff} of {n} differ)");
        }
        let prob_max = prob.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        for &idx in &chosen {
            prob[idx] = prob_max;
        }
        let mut best = 0usize;
        let mut min = f32::INFINITY;
        for (ii, &x) in prob.iter().enumerate() {
            if x < min {
                min = x;
                best = ii;
            }
        }
        while chosen.contains(&best) {
            prob[best] = prob_max;
            best = 0usize;
            min = f32::INFINITY;
            for (ii, &x) in prob.iter().enumerate() {
                if x < min {
                    min = x;
                    best = ii;
                }
            }
        }
        chosen.insert(best);
        chosen_order.push(best);
        y2.set_col(i, z_norm.col(best));
    }
    // Compare the full chosen-index list against the C++ reference.
    {
        let cpp_text = std::fs::read_to_string("/tmp/opencode/cpp_chosen_list.txt").unwrap();
        let cpp: Vec<usize> = cpp_text
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        let mut first_diff = None;
        for i in 0..cpp.len().min(chosen_order.len()) {
            if cpp[i] != chosen_order[i] {
                first_diff = Some(i);
                break;
            }
        }
        println!(
            "chosen list: rust len={} cpp len={} first diff at {:?} (rust {}, cpp {})",
            chosen_order.len(),
            cpp.len(),
            first_diff,
            first_diff.map(|i| chosen_order[i]).unwrap_or(0),
            first_diff.map(|i| cpp[i]).unwrap_or(0),
        );
    }
    compare_stage("stage2", &y2, "/tmp/opencode/cpp_stage2_y.bin");

    // ---- Stage 3: which picks were chosen (compare chosen sets via columns) ----
    // Print the first 10 chosen indices from stage 2 by matching columns to
    // z_norm columns is expensive; instead compare stage3 output.
    let mut y3 = y2.clone();
    for _ in 0..10 {
        // inline single km_iterate pass (mirrors the crate's port)
        let n_dims = y3.nrows;
        let n_gaus = y3.ncols;
        let mut acc_means = vec![0.0f32; n_dims * n_gaus];
        let mut acc_hefts = vec![0u32; n_gaus];
        let mut last_indx = vec![0usize; n_gaus];
        let old_means = y3.clone();
        for i in 0..n {
            let x_col = z_norm.col(i);
            let mut min_dist = f32::INFINITY;
            let mut best_g = 0usize;
            for g in 0..n_gaus {
                // two-pass paired squared distance
                let m = old_means.col(g);
                let mut acc1 = 0.0f32;
                let mut acc2 = 0.0f32;
                let mut ii = 0usize;
                let mut jj = 1usize;
                while jj < n_dims {
                    let ti = x_col[ii] - m[ii];
                    let tj = x_col[jj] - m[jj];
                    acc1 += ti * ti;
                    acc2 += tj * tj;
                    ii += 2;
                    jj += 2;
                }
                if ii < n_dims {
                    let ti = x_col[ii] - m[ii];
                    acc1 += ti * ti;
                }
                let dist = acc1 + acc2;
                if dist < min_dist {
                    min_dist = dist;
                    best_g = g;
                }
            }
            for dd in 0..n_dims {
                acc_means[best_g * n_dims + dd] += x_col[dd];
            }
            acc_hefts[best_g] += 1;
            last_indx[best_g] = i;
        }
        for g in 0..n_gaus {
            let heft = acc_hefts[g];
            for dd in 0..n_dims {
                let v = if heft >= 1 {
                    acc_means[g * n_dims + dd] / heft as f32
                } else {
                    0.0
                };
                y3.set(dd, g, v);
            }
        }
    }
    compare_stage("stage3", &y3, "/tmp/opencode/cpp_stage3_y.bin");
}
