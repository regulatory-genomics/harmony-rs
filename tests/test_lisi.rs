//! LISI golden comparison against the reference fixture (kd-tree tolerance
//! atol=0.01).

mod common;

use common::read_tsv;
use harmony::compute_lisi;
use std::collections::HashMap;

fn data_path(name: &str) -> String {
    format!("{}/data/{}", env!("CARGO_MANIFEST_DIR"), name)
}

#[test]
fn test_lisi() {
    let (x_header, x_cols) = read_tsv(&data_path("lisi_x.tsv.gz"));
    let (_, metadata_cols) = read_tsv(&data_path("lisi_metadata.tsv.gz"));
    let label_colnames: Vec<String> = metadata_cols.keys().cloned().collect();
    let perplexity = 30.0;

    // X: N x d row-major from numeric columns.
    let mut numeric: Vec<Vec<f64>> = Vec::new();
    for name in &x_header {
        if let Some(common::Column::Float(v)) = x_cols.get(name) {
            numeric.push(v.clone());
        }
    }
    let n = numeric.first().map(|v| v.len()).unwrap_or(0);
    let d = numeric.len();
    let mut x = vec![0.0f64; n * d];
    for (j, col) in numeric.iter().enumerate() {
        for (i, &v) in col.iter().enumerate() {
            x[i * d + j] = v;
        }
    }

    let metadata: HashMap<String, Vec<String>> = metadata_cols
        .into_iter()
        .map(|(k, v)| {
            let strings = match v {
                common::Column::String(s) => s,
                common::Column::Float(f) => f.iter().map(|x| x.to_string()).collect(),
            };
            (k, strings)
        })
        .collect();

    // lisi: N x n_labels (row-major with n_labels per row).
    let lisi = compute_lisi(&x, n, d, &metadata, &label_colnames, perplexity)
        .expect("compute_lisi failed");

    // Reference fixture.
    let (_, lisi_cols) = read_tsv(&data_path("lisi_lisi.tsv.gz"));
    let lisi_test: Vec<Vec<f64>> = label_colnames
        .iter()
        .map(|name| lisi_cols[name].floats().to_vec())
        .collect();

    // Compare (atol=0.01 — accounts for kd-tree differences).
    for (li, lt) in lisi_test.iter().enumerate() {
        for i in 0..n {
            let actual = lisi[i * label_colnames.len() + li];
            let expected = lt[i];
            assert!(
                (actual - expected).abs() <= 0.01,
                "mismatch at {i}, label {li}: actual {actual}, expected {expected}"
            );
        }
    }
}
