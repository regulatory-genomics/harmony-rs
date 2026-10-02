//! Debug comparison against the Python oracle intermediate state dumped to
//! /tmp/opencode/py_ref_small_iter1.json (pbmc_3500 + donor,
//! max_iter_harmony=1, max_iter_kmeans=1, defaults elsewhere).
//!
//! Run with: cargo test --test debug_oracle -- --ignored --nocapture

#![allow(clippy::needless_range_loop)]
mod common;

use common::{cols_to_matrix, read_tsv};
use harmony::{run_harmony, RunHarmonyArgs};
use serde_json::Value;
use std::collections::HashMap;

fn data_path(name: &str) -> String {
    format!("{}/data/{}", env!("CARGO_MANIFEST_DIR"), name)
}

fn load_pcs() -> (Vec<f64>, usize, usize, HashMap<String, Vec<String>>) {
    let (_, meta_cols) = read_tsv(&data_path("pbmc_3500_meta.tsv.gz"));
    let (pcs_header, pcs_cols) = read_tsv(&data_path("pbmc_3500_pcs.tsv.gz"));
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
    (data_mat, n, d, meta_map)
}

fn max_abs_diff(label: &str, a: &[f64], b: &[f64]) -> (usize, f64) {
    assert_eq!(
        a.len(),
        b.len(),
        "{label}: length mismatch {} vs {}",
        a.len(),
        b.len()
    );
    let mut worst = 0usize;
    let mut maxd = 0.0f64;
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs();
        if d > maxd {
            maxd = d;
            worst = i;
        }
    }
    println!("{label}: max abs diff = {maxd:.3e} at index {worst}");
    (worst, maxd)
}

#[test]
#[ignore]
fn compare_iter1_state() {
    let (data_mat, n, d, meta_map) = load_pcs();

    let result = run_harmony(&RunHarmonyArgs {
        max_iter_harmony: 1,
        max_iter_kmeans: 1,
        ..RunHarmonyArgs::python_defaults(&data_mat, n, d, &meta_map, vec!["donor".to_string()])
    })
    .expect("run_harmony failed");

    let raw = std::fs::read_to_string("/tmp/opencode/py_ref_small_iter1.json").unwrap();
    let py: Value = serde_json::from_str(&raw).unwrap();

    // Parse r early (needed for the Python-side entropy).
    let py_r: Vec<Vec<f64>> = py["r"]
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

    // --- objectives ---
    let py_obj_k: Vec<f64> = py["objective_kmeans"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let py_obj_h: Vec<f64> = py["objective_harmony"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    println!("rust objective_kmeans: {:?}", result.objective_kmeans);
    println!("py   objective_kmeans: {py_obj_k:?}");
    println!("rust objective_harmony: {:?}", result.objective_harmony);
    println!("py   objective_harmony: {py_obj_h:?}");
    println!("rust dist:    {:?}", result.objective_kmeans_dist);
    println!("rust entropy: {:?}", result.objective_kmeans_entropy);
    println!("rust cross:   {:?}", result.objective_kmeans_cross);
    // Python-side entropy from r (sigma = 0.1 uniform, norm_const = 2000/N).
    let norm_const = 2000.0 / n as f64;
    let mut py_entropy_raw = 0.0f64;
    for j in 0..n {
        for g in 0..result.k {
            let v = py_r[j][g];
            if v > 0.0 {
                py_entropy_raw += v * v.ln();
            }
        }
    }
    let py_entropy = py_entropy_raw * 0.1 * norm_const;
    println!("py entropy (from r, sigma=0.1): {py_entropy:?}");
    let py_dist_cross: Vec<f64> = py_obj_k
        .iter()
        .zip(&result.objective_kmeans_entropy)
        .map(|(t, e)| t - e)
        .collect();
    println!("py dist+cross (total - entropy): {py_dist_cross:?}");
    max_abs_diff("objective_kmeans", &result.objective_kmeans, &py_obj_k);
    max_abs_diff("objective_harmony", &result.objective_harmony, &py_obj_h);
    println!(
        "kmeans_rounds rust={:?} py={:?}",
        result.kmeans_rounds,
        py["kmeans_rounds"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect::<Vec<_>>()
    );

    // --- z_corr: py is N x d; rust internal is d x N column-major ---
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
    let mut py_z_flat = vec![0.0f64; n * d];
    for j in 0..n {
        for i in 0..d {
            py_z_flat[j * d + i] = py_z[j][i];
        }
    }
    max_abs_diff("z_corr", &result.z_corr, &py_z_flat);

    // --- y: py is d x K; rust internal y is d x K column-major ---
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
    let mut py_y_flat = vec![0.0f64; d * k];
    for i in 0..d {
        for g in 0..k {
            py_y_flat[g * d + i] = py_y[i][g];
        }
    }
    max_abs_diff("y", &result.y, &py_y_flat);

    // --- r: py is N x K; rust internal r is K x N column-major ---
    let mut py_r_flat = vec![0.0f64; n * k];
    for j in 0..n {
        for g in 0..k {
            py_r_flat[j * k + g] = py_r[j][g];
        }
    }
    max_abs_diff("r", &result.r, &py_r_flat);
}
