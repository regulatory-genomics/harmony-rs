//! Benchmark harness: time `run_harmony` only (datasets parsed once, before
//! timing) on the pbmc3k (small) and ircolitis (medium) configs.
//!
//! Run with: cargo test --release --test bench -- --ignored --nocapture

mod common;

use common::{cols_to_matrix, read_tsv};
use harmonypy::{run_harmony, RunHarmonyArgs};
use std::collections::HashMap;
use std::time::Instant;

fn data_path(name: &str) -> String {
    format!("{}/data/{}", env!("CARGO_MANIFEST_DIR"), name)
}

fn meta_map(meta_cols: &HashMap<String, common::Column>) -> HashMap<String, Vec<String>> {
    let mut map = HashMap::new();
    for (name, col) in meta_cols {
        match col {
            common::Column::String(v) => {
                map.insert(name.clone(), v.clone());
            }
            common::Column::Float(v) => {
                map.insert(name.clone(), v.iter().map(|x| x.to_string()).collect());
            }
        }
    }
    map
}

#[allow(clippy::too_many_arguments)]
fn bench(
    label: &str,
    data_mat: &[f64],
    n: usize,
    d: usize,
    meta: &HashMap<String, Vec<String>>,
    vars: Vec<String>,
    max_iter_harmony: usize,
    runs: usize,
) {
    let mut times = Vec::new();
    let mut rounds = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let result = run_harmony(&RunHarmonyArgs {
            max_iter_harmony,
            ..RunHarmonyArgs::python_defaults(data_mat, n, d, meta, vars.clone())
        })
        .expect("run_harmony failed");
        times.push(start.elapsed().as_secs_f64());
        rounds = result.kmeans_rounds.clone();
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = times[0];
    let median = times[times.len() / 2];
    println!("{label}: min={min:.3}s median={median:.3}s over {runs} runs (rounds={rounds:?})");
}

#[test]
#[ignore]
fn bench_small_pbmc3k() {
    let (_, meta_cols) = read_tsv(&data_path("pbmc_3500_meta.tsv.gz"));
    let (pcs_header, pcs_cols) = read_tsv(&data_path("pbmc_3500_pcs.tsv.gz"));
    let (data_mat, n, d) = cols_to_matrix(&pcs_cols, &pcs_header);
    let meta = meta_map(&meta_cols);
    bench(
        "small pbmc3k (3500 cells, 10 iters)",
        &data_mat,
        n,
        d,
        &meta,
        vec!["donor".to_string()],
        10,
        5,
    );
}

#[test]
#[ignore]
fn bench_medium_ircolitis() {
    let (_, meta_cols) = read_tsv(&data_path("ircolitis_blood_cd8_obs.tsv.gz"));
    let (pcs_header, pcs_cols) = read_tsv(&data_path("ircolitis_blood_cd8_pcs.tsv.gz"));
    let (data_mat, n, d) = cols_to_matrix(&pcs_cols, &pcs_header);
    let meta = meta_map(&meta_cols);
    bench(
        "medium ircolitis (68785 cells, 1 iter)",
        &data_mat,
        n,
        d,
        &meta,
        vec!["batch".to_string()],
        1,
        5,
    );
}
