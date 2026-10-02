//! Medium-gate calibration: per-PC corr and relative diffs vs the Python
//! golden (differential_medium.json).
//!
//! Run with: cargo test --test debug_oracle -- --ignored --nocapture

#![allow(clippy::needless_range_loop)]
mod common;

use common::{cols_to_matrix, pearsonr, read_tsv};
use harmony::{run_harmony, RunHarmonyArgs};
use serde_json::Value;
use std::collections::HashMap;

fn data_path(name: &str) -> String {
    format!("{}/data/{}", env!("CARGO_MANIFEST_DIR"), name)
}

#[test]
#[ignore]
fn calibrate_medium_gates() {
    let (_, meta_cols) = read_tsv(&data_path("ircolitis_blood_cd8_obs.tsv.gz"));
    let (pcs_header, pcs_cols) = read_tsv(&data_path("ircolitis_blood_cd8_pcs.tsv.gz"));
    let (data_mat, n, d) = cols_to_matrix(&pcs_cols, &pcs_header);

    let mut meta_map: HashMap<String, Vec<String>> = HashMap::new();
    for (name, col) in &meta_cols {
        match col {
            common::Column::String(v) => {
                meta_map.insert(name.clone(), v.clone());
            }
            common::Column::Float(v) => {
                meta_map.insert(name.clone(), v.iter().map(|x| x.to_string()).collect());
            }
        }
    }

    let result = run_harmony(&RunHarmonyArgs {
        max_iter_harmony: 1,
        ..RunHarmonyArgs::python_defaults(&data_mat, n, d, &meta_map, vec!["batch".to_string()])
    })
    .expect("run_harmony failed");

    let raw = std::fs::read_to_string(format!(
        "{}/tests/fixtures/differential_medium.json",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    let py: Value = serde_json::from_str(&raw).unwrap();

    let py_z: Vec<Vec<f64>> = py["z_corr"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect()
        })
        .collect();

    for i in 0..d {
        let mut res_pc = Vec::with_capacity(n);
        let mut py_pc = Vec::with_capacity(n);
        for j in 0..n {
            res_pc.push(result.z_corr[j * d + i]);
            py_pc.push(py_z[j][i]);
        }
        let c = pearsonr(&res_pc, &py_pc);
        let maxd = res_pc
            .iter()
            .zip(&py_pc)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        let scale = py_pc.iter().map(|b| b.abs()).fold(0.0f64, f64::max);
        println!(
            "PC{:2}: corr={:.6} max_abs_diff={:.3e} max_|py|={:.3e} rel={:.3e}",
            i + 1,
            c,
            maxd,
            scale,
            maxd / scale
        );
    }

    // y per-cluster
    let py_y: Vec<Vec<f64>> = py["y"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect()
        })
        .collect();
    let k = result.k;
    let mut max_y = 0.0f64;
    let mut scale_y = 0.0f64;
    for i in 0..d {
        for g in 0..k {
            max_y = max_y.max((result.y[g * d + i] - py_y[i][g]).abs());
            scale_y = scale_y.max(py_y[i][g].abs());
        }
    }
    println!(
        "y: max_abs_diff={:.3e} max_|py|={:.3e} rel={:.3e}",
        max_y,
        scale_y,
        max_y / scale_y
    );

    // objectives
    let py_obj: Vec<f64> = py["objective_kmeans"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    for (i, (a, b)) in result.objective_kmeans.iter().zip(&py_obj).enumerate() {
        println!(
            "obj[{i}]: rust={a:.4} py={b:.4} rel={:.3e}",
            (a - b).abs() / b.abs()
        );
    }

    // --- init-state f64 reference (max_iter_harmony=0: y/r/z_corr are the
    // init state). This validates the objective *formula* (not the f32
    // summation order): the f64 total lands within ~3e-4 of the Python
    // golden, while the Rust f32 dual-accumulator is ~2.4e-3 away — the
    // residual is f32 accumulation error over the N*K=6.9M-element sums
    // (the C++ backend sums these via OpenBLAS sdot, which is more
    // accurate but bitwise unreplicable; see docs/parity.md).
    let result0 = run_harmony(&RunHarmonyArgs {
        max_iter_harmony: 0,
        ..RunHarmonyArgs::python_defaults(&data_mat, n, d, &meta_map, vec!["batch".to_string()])
    })
    .expect("run_harmony(0) failed");

    // batch proportions (Pr_b) and sizes from the metadata, sorted-unique codes.
    let labels = meta_map.get("batch").unwrap();
    let mut uniq: Vec<String> = labels.to_vec();
    uniq.sort();
    uniq.dedup();
    let b = uniq.len();
    let mut pr_b = vec![0.0f64; b];
    let mut o_f64 = vec![0.0f64; b * result0.k];
    let mut row_sums = vec![0.0f64; result0.k];
    for (j, lab) in labels.iter().enumerate() {
        let bi = uniq.iter().position(|u| u == lab).unwrap();
        pr_b[bi] += 1.0;
        for kk in 0..result0.k {
            let r = result0.r[j * result0.k + kk];
            o_f64[bi * result0.k + kk] += r;
            row_sums[kk] += r;
        }
    }
    for v in pr_b.iter_mut() {
        *v /= n as f64;
    }

    let mut dist_ref = 0.0f64;
    let mut ent_ref = 0.0f64;
    let k0 = result0.k;
    for kk in 0..k0 {
        for j in 0..n {
            let r = result0.r[j * k0 + kk];
            let mut dot = 0.0f64;
            for dd in 0..d {
                dot += result0.y[kk * d + dd] * result0.z_corr[j * d + dd];
            }
            dist_ref += r * (2.0 * (1.0 - dot));
            if r > 0.0 {
                ent_ref += r * r.ln() * 0.1;
            }
        }
    }
    let mut cross_ref = 0.0f64;
    for bi in 0..b {
        for kk in 0..k0 {
            let o = o_f64[bi * k0 + kk];
            let e = row_sums[kk] * pr_b[bi];
            let ratio = ((o + e + 1.0) / (2.0 * e + 1.0)).ln();
            cross_ref += ratio * 2.0 * 0.1 * o;
        }
    }
    let total_f64 = (dist_ref + ent_ref + cross_ref) * (2000.0 / n as f64);
    let total_rust = result0.objective_kmeans[0];
    let total_py = py_obj[0];
    println!("init-state (max_iter_harmony=0):");
    println!(
        "  f64  dist = {dist_ref:.4}   rust = {:.4}",
        result0.objective_kmeans_dist[0] / (2000.0 / n as f64)
    );
    println!(
        "  f64  ent  = {ent_ref:.4}   rust = {:.4}",
        result0.objective_kmeans_entropy[0] / (2000.0 / n as f64)
    );
    println!(
        "  f64  cross= {cross_ref:.4} rust = {:.4}",
        result0.objective_kmeans_cross[0] / (2000.0 / n as f64)
    );
    println!("  total: f64 = {total_f64:.4}  rust = {total_rust:.4}  py = {total_py:.4}");
    println!(
        "  f64-vs-py rel = {:.3e}   rust-vs-py rel = {:.3e}",
        (total_f64 - total_py).abs() / total_py,
        (total_rust - total_py).abs() / total_py
    );
}
