//! Debug bisect for the medium (ircolitis + batch) divergence: run Rust with
//! max_iter_kmeans = 1, 2, 3 and compare objectives + z_corr against the
//! Python oracle dumps in /tmp/opencode/py_bisect_medium_km{N}.json.
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

#[test]
#[ignore]
fn bisect_medium_kmeans() {
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

    for kmeans in [1usize, 2, 3] {
        let result = run_harmony(&RunHarmonyArgs {
            max_iter_harmony: 1,
            max_iter_kmeans: kmeans,
            ..RunHarmonyArgs::python_defaults(&data_mat, n, d, &meta_map, vec!["batch".to_string()])
        })
        .expect("run_harmony failed");

        let raw =
            std::fs::read_to_string(format!("/tmp/opencode/py_bisect_medium_km{kmeans}.json"))
                .unwrap();
        let py: Value = serde_json::from_str(&raw).unwrap();
        let py_obj: Vec<f64> = py["objective_kmeans"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        println!(
            "kmeans={kmeans} rounds rust={:?} py={:?}",
            result.kmeans_rounds,
            py["kmeans_rounds"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap())
                .collect::<Vec<_>>()
        );
        for (i, (a, b)) in result.objective_kmeans.iter().zip(&py_obj).enumerate() {
            println!("  obj[{i}]: rust={a:.4} py={b:.4} diff={:.4}", a - b);
        }
        // z_corr diff
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
        let mut max_z = 0.0f64;
        for j in 0..n {
            for i in 0..d {
                max_z = max_z.max((result.z_corr[j * d + i] - py_z[j][i]).abs());
            }
        }
        println!("  z_corr max abs diff = {max_z:.3e}");
    }
}
