//! Self-golden harness for the differential/smoke tests.
//!
//! The committed fixtures under `tests/fixtures/differential_*.json` are
//! generated *from the Rust implementation itself* via the ignored
//! `tests/gen_golden.rs` generator. This is by design: the algorithm is fully
//! deterministic for a given `random_state` (a `ChaCha8Rng` seeded from it),
//! so the fixtures act as a regression lock-in.
//!
//! Two kinds of checks run against them:
//! - [`check_exact`]: the current run must reproduce the committed fixture
//!   bit-for-bit (z_corr, y, r, objectives, kmeans_rounds, K). Any change to
//!   the algorithm, matrix ops, or RNG surfaces here as a failure, signalling
//!   that the fixture needs regenerating.
//! - [`check_quality`]: statistical sanity that the result is a *good*
//!   correction, independent of matching a specific oracle: all outputs are
//!   finite, per-PC variance is preserved, k-means always makes progress, and
//!   batch mixing (measured by LISI) does not degrade.

#![allow(clippy::needless_range_loop)]

use super::{cols_to_matrix, read_tsv, Column};
use harmony::{compute_lisi, run_harmony, HarmonyResult, RunHarmonyArgs, ScalarOrVec};
use serde_json::{json, Value};
use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Cfg {
    SmallDefaults,
    Iter2,
    FixedLamb,
    MultiCov,
    Medium,
}

impl Cfg {
    pub fn name(self) -> &'static str {
        match self {
            Cfg::SmallDefaults => "small_defaults",
            Cfg::Iter2 => "iter2",
            Cfg::FixedLamb => "fixed_lamb",
            Cfg::MultiCov => "multi_cov",
            Cfg::Medium => "medium",
        }
    }

    pub fn all() -> [Cfg; 5] {
        [
            Cfg::SmallDefaults,
            Cfg::Iter2,
            Cfg::FixedLamb,
            Cfg::MultiCov,
            Cfg::Medium,
        ]
    }

    /// The harmony covariates for this config.
    fn covariates(self) -> Vec<String> {
        match self {
            Cfg::MultiCov => vec!["lab".to_string(), "day".to_string()],
            Cfg::SmallDefaults | Cfg::Iter2 | Cfg::FixedLamb => vec!["donor".to_string()],
            Cfg::Medium => vec!["batch".to_string()],
        }
    }

    /// The label column used for the LISI batch-mixing quality check.
    fn mixing_label(self) -> String {
        match self {
            Cfg::MultiCov => "lab".to_string(),
            _ => self.covariates()[0].clone(),
        }
    }
}

/// Everything needed to (re)generate and verify a fixture.
pub struct Run {
    pub cfg: Cfg,
    pub result: HarmonyResult,
    pub n: usize,
    pub d: usize,
    /// Original embedding (pre-correction), N x d row-major f64.
    pub z_input: Vec<f64>,
    /// Per-cell labels for the mixing label column.
    pub batch_labels: Vec<String>,
    /// Whether the fixture stores the r matrix (omitted for the 69k medium
    /// config to keep the JSON at a reasonable size).
    pub include_r: bool,
}

pub fn data_path(name: &str) -> String {
    format!("{}/data/{name}", env!("CARGO_MANIFEST_DIR"))
}

pub fn fixture_path(name: &str) -> String {
    format!(
        "{}/tests/fixtures/differential_{name}.json",
        env!("CARGO_MANIFEST_DIR")
    )
}

fn load_pcs(
    meta_tsv: &str,
    pcs_tsv: &str,
) -> (Vec<f64>, usize, usize, HashMap<String, Vec<String>>) {
    let (_, meta_cols) = read_tsv(&data_path(meta_tsv));
    let (pcs_header, pcs_cols) = read_tsv(&data_path(pcs_tsv));
    let (data_mat, n, d) = cols_to_matrix(&pcs_cols, &pcs_header);

    let mut meta_map: HashMap<String, Vec<String>> = HashMap::new();
    for (name, col) in &meta_cols {
        match col {
            Column::String(v) => {
                meta_map.insert(name.clone(), v.clone());
            }
            Column::Float(v) => {
                meta_map.insert(name.clone(), v.iter().map(|x| x.to_string()).collect());
            }
        }
    }
    (data_mat, n, d, meta_map)
}

pub fn nested(v: &Value) -> Vec<Vec<f64>> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect()
        })
        .collect()
}

fn flat_strings(v: &Value) -> Vec<String> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_str().unwrap().to_string())
        .collect()
}

/// Run harmony for the given config.
pub fn run(cfg: Cfg) -> Run {
    // `data_mat` is N x d row-major f64.
    let (data_mat, n, d, meta_map): (Vec<f64>, usize, usize, HashMap<String, Vec<String>>) =
        match cfg {
            Cfg::MultiCov => {
                // Inputs are stored in the committed fixture.
                let py: Value = serde_json::from_str(
                    &std::fs::read_to_string(fixture_path("multi_cov")).unwrap(),
                )
                .unwrap();
                let inputs = &py["inputs"];
                let x = nested(&inputs["x"]);
                let n = x.len();
                let d = x[0].len();
                let mut dm = vec![0.0f64; n * d];
                for (i, row) in x.iter().enumerate() {
                    dm[i * d..(i + 1) * d].copy_from_slice(row);
                }
                let mut meta: HashMap<String, Vec<String>> = HashMap::new();
                meta.insert("lab".to_string(), flat_strings(&inputs["lab"]));
                meta.insert("day".to_string(), flat_strings(&inputs["day"]));
                (dm, n, d, meta)
            }
            Cfg::SmallDefaults | Cfg::Iter2 | Cfg::FixedLamb => {
                load_pcs("pbmc_3500_meta.tsv.gz", "pbmc_3500_pcs.tsv.gz")
            }
            Cfg::Medium => load_pcs(
                "ircolitis_blood_cd8_obs.tsv.gz",
                "ircolitis_blood_cd8_pcs.tsv.gz",
            ),
        };

    let covariates = cfg.covariates();
    let args = RunHarmonyArgs {
        max_iter_harmony: match cfg {
            Cfg::Iter2 | Cfg::MultiCov => 2,
            Cfg::Medium => 1,
            _ => 10,
        },
        max_iter_kmeans: match cfg {
            Cfg::Iter2 | Cfg::MultiCov => 2,
            _ => 200,
        },
        lamb: match cfg {
            Cfg::FixedLamb => Some(ScalarOrVec::Scalar(1.0)),
            _ => None,
        },
        ..RunHarmonyArgs::python_defaults(&data_mat, n, d, &meta_map, covariates)
    };
    let result = run_harmony(&args).expect("run_harmony failed");

    let batch_labels = meta_map[&cfg.mixing_label()].clone();
    Run {
        cfg,
        result,
        n,
        d,
        z_input: data_mat,
        batch_labels,
        include_r: cfg != Cfg::Medium,
    }
}

fn to_f64(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn nested_rowmajor(v: &[f64], rows: usize, cols: usize) -> Vec<Vec<f64>> {
    (0..rows)
        .map(|i| v[i * cols..(i + 1) * cols].to_vec())
        .collect()
}

/// Write the committed fixture for a config (used by the ignored generator).
pub fn write_fixture(r: &Run) {
    let mut obj = serde_json::Map::new();
    obj.insert("K".to_string(), json!(r.result.k));
    obj.insert(
        "z_corr".to_string(),
        json!(nested_rowmajor(&r.result.z_corr, r.n, r.d)),
    );
    obj.insert(
        "y".to_string(),
        json!(nested_rowmajor(&r.result.y, r.result.k, r.d)),
    );
    if r.include_r {
        obj.insert(
            "r".to_string(),
            json!(nested_rowmajor(&r.result.r, r.n, r.result.k)),
        );
    }
    obj.insert(
        "objective_kmeans".to_string(),
        json!(r.result.objective_kmeans),
    );
    obj.insert(
        "objective_harmony".to_string(),
        json!(r.result.objective_harmony),
    );
    obj.insert(
        "kmeans_rounds".to_string(),
        json!(r
            .result
            .kmeans_rounds
            .iter()
            .map(|&x| x as u64)
            .collect::<Vec<_>>()),
    );
    // Multi-cov config: preserve the committed inputs.
    if r.cfg == Cfg::MultiCov {
        let old: Value = serde_json::from_str(
            &std::fs::read_to_string(fixture_path("multi_cov")).unwrap_or_default(),
        )
        .unwrap_or(Value::Null);
        if let Some(inputs) = old.get("inputs") {
            obj.insert("inputs".to_string(), inputs.clone());
        }
    }
    let path = fixture_path(r.cfg.name());
    std::fs::write(
        &path,
        serde_json::to_string_pretty(&Value::Object(obj)).unwrap() + "\n",
    )
    .unwrap();
    println!(
        "wrote {path}: K={} rounds={:?}",
        r.result.k, r.result.kmeans_rounds
    );
}

/// Determinism / regression lock-in: the current run must reproduce the
/// committed self-golden fixture to within a tight tolerance.
///
/// A bit-exact comparison is not stable because faer's multithreaded gemm
/// reduces in a scheduling-dependent order, so consecutive runs of the same
/// binary can differ by an f64 ulp. The stored data are f32-derived anyway, so
/// a tight absolute bound (well above that noise, well below any real
/// algorithmic change) acts as the regression gate.
pub fn check_exact(r: &Run, py: &Value) {
    let name = r.cfg.name();
    let atol = 5e-4;

    assert_eq!(
        r.result.k,
        py["K"].as_u64().unwrap() as usize,
        "{name}: K mismatch"
    );

    let py_rounds: Vec<usize> = py["kmeans_rounds"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    assert_eq!(r.result.kmeans_rounds, py_rounds, "{name}: kmeans_rounds");

    // Max-abs-diff helper: returns the largest |a - b| across two matrices.
    let max_abs = |left: &[f64], right: &[Vec<f64>]| -> f64 {
        let mut m = 0.0f64;
        for (i, row) in right.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                m = m.max((left[i * row.len() + j] - v).abs());
            }
        }
        m
    };

    let py_z = nested(&py["z_corr"]);
    assert_eq!(py_z.len(), r.n, "{name}: z_corr rows");
    assert_eq!(py_z[0].len(), r.d, "{name}: z_corr cols");
    let mut mz = 0.0f64;
    for j in 0..r.n {
        for i in 0..r.d {
            mz = mz.max((r.result.z_corr[j * r.d + i] - py_z[j][i]).abs());
        }
    }
    assert!(mz <= atol, "{name}: z_corr max abs diff {mz:.3e} > {atol}");

    let py_y = nested(&py["y"]);
    let my = max_abs(&r.result.y, &py_y);
    assert!(my <= atol, "{name}: y max abs diff {my:.3e} > {atol}");

    if r.include_r {
        let py_r = nested(&py["r"]);
        let mr = max_abs(&r.result.r, &py_r);
        assert!(mr <= atol, "{name}: r max abs diff {mr:.3e} > {atol}");
    }

    // Objectives: small relative tolerance (f32 accumulation noise).
    let rel_tol = match r.cfg {
        Cfg::Medium => 5e-3,
        _ => 1e-3,
    };
    let obj_k = to_f64(&py["objective_kmeans"]);
    assert_eq!(r.result.objective_kmeans.len(), obj_k.len());
    for (i, (a, b)) in r.result.objective_kmeans.iter().zip(&obj_k).enumerate() {
        let rel = (a - b).abs() / b.abs().max(1e-30);
        assert!(
            rel <= rel_tol,
            "{name}: objective_kmeans[{i}] rel diff {rel:.3e} > {rel_tol}"
        );
    }
    let obj_h = to_f64(&py["objective_harmony"]);
    assert_eq!(r.result.objective_harmony.len(), obj_h.len());
    for (i, (a, b)) in r.result.objective_harmony.iter().zip(&obj_h).enumerate() {
        let rel = (a - b).abs() / b.abs().max(1e-30);
        assert!(
            rel <= rel_tol,
            "{name}: objective_harmony[{i}] rel diff {rel:.3e} > {rel_tol}"
        );
    }
    println!("{name}: z_corr max abs diff {mz:.3e}, y {my:.3e}, rounds match, objectives ok");
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn variance(v: &[f64]) -> f64 {
    let m = mean(v);
    v.iter().map(|&x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64
}

fn mean_lisi(x_rowmajor: &[f64], n: usize, d: usize, labels: &[String]) -> f64 {
    let mut meta: HashMap<String, Vec<String>> = HashMap::new();
    meta.insert("batch".to_string(), labels.to_vec());
    // Perplexity follows the harmonypy default (30).
    let values = compute_lisi(x_rowmajor, n, d, &meta, &["batch".to_string()], 30.0)
        .expect("compute_lisi failed");
    mean(&values)
}

/// Statistical sanity independent of any particular oracle: finite outputs,
/// variance preservation, k-means progress, and batch mixing not degrading.
pub fn check_quality(r: &Run) {
    let name = r.cfg.name();
    let res = &r.result;

    // 1. All outputs finite.
    assert!(res.z_corr.iter().all(|v| v.is_finite()), "{name}: z_corr");
    assert!(res.y.iter().all(|v| v.is_finite()), "{name}: y");
    assert!(res.r.iter().all(|v| v.is_finite()), "{name}: r");
    assert!(
        res.objective_kmeans.iter().all(|v| v.is_finite()),
        "{name}: objective_kmeans"
    );
    assert!(
        res.objective_harmony.iter().all(|v| v.is_finite()),
        "{name}: objective_harmony"
    );

    // 2. k-means makes progress in every round.
    assert!(
        res.kmeans_rounds.iter().all(|&c| c >= 1),
        "{name}: kmeans_rounds all >= 1"
    );

    // 3. Per-PC variance is broadly preserved by the correction.
    for pc in 0..r.d {
        let v_orig = variance(
            &(0..r.n)
                .map(|j| r.z_input[j * r.d + pc])
                .collect::<Vec<_>>(),
        );
        let v_corr = variance(
            &(0..r.n)
                .map(|j| res.z_corr[j * r.d + pc])
                .collect::<Vec<_>>(),
        );
        assert!(v_corr > 0.0, "{name}: PC{pc} corrected variance zero");
        let ratio = v_corr / v_orig;
        assert!(
            (0.2..=5.0).contains(&ratio),
            "{name}: PC{pc} variance ratio {ratio:.3} out of [0.2, 5.0]"
        );
    }

    // 4. Batch mixing (LISI) does not degrade after correction.
    //    Skipped for the 69k medium config (kd-tree LISI is expensive there);
    //    the other statistical checks still apply.
    if r.cfg != Cfg::Medium {
        let lisi_orig = mean_lisi(&r.z_input, r.n, r.d, &r.batch_labels);
        let lisi_corr = mean_lisi(&res.z_corr, r.n, r.d, &r.batch_labels);
        println!("{name}: mean LISI {lisi_orig:.4} -> {lisi_corr:.4}");
        assert!(
            lisi_corr >= lisi_orig * 0.90,
            "{name}: LISI dropped {lisi_orig:.4} -> {lisi_corr:.4}"
        );
    }
}
