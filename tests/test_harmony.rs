//! Behavioral parity with the reference test suite.

mod common;

use common::{cols_to_matrix, pearsonr, read_tsv, Column};
use harmony::harmony::core::{objective_converged, HarmonyError};
use harmony::{run_harmony, RunHarmonyArgs, ScalarOrVec};
use std::collections::HashMap;

fn data_path(name: &str) -> String {
    format!("{}/data/{}", env!("CARGO_MANIFEST_DIR"), name)
}

/// Port of `test_run_harmony_small` / `run_harmony` helper: compare against
/// the R harmony2 reference (per-PC correlation >= min_correlation).
fn run_harmony_golden(
    meta_tsv: &str,
    pcs_tsv: &str,
    harmonized_tsv: &str,
    batch_var: &str,
    min_correlation: f64,
) {
    let (_, meta_cols) = read_tsv(&data_path(meta_tsv));
    let (pcs_header, pcs_cols) = read_tsv(&data_path(pcs_tsv));
    let (data_mat, n, d) = cols_to_matrix(&pcs_cols, &pcs_header);

    let mut meta_map: HashMap<String, Vec<String>> = HashMap::new();
    for (name, col) in &meta_cols {
        if let Column::String(v) = col {
            meta_map.insert(name.clone(), v.clone());
        } else if let Column::Float(v) = col {
            meta_map.insert(name.clone(), v.iter().map(|x| x.to_string()).collect());
        }
    }

    let ho = run_harmony(&RunHarmonyArgs::python_defaults(
        &data_mat,
        n,
        d,
        &meta_map,
        vec![batch_var.to_string()],
    ))
    .expect("run_harmony failed");

    assert_eq!(ho.k, (n as f64 / 30.0).round().min(100.0) as usize);

    // Compare to expected results from R.
    let res = &ho.z_corr; // d x N (column-major, d = PCs)
    let (harm_header, harm_cols) = read_tsv(&data_path(harmonized_tsv));
    let (harm, _, _) = cols_to_matrix(&harm_cols, &harm_header);

    let n_pcs = d;
    let mut cors_values = Vec::new();
    for i in 0..n_pcs {
        // res column i (PC i for each cell) vs harm column i.
        let mut res_pc = Vec::with_capacity(n);
        let mut harm_pc = Vec::with_capacity(n);
        for j in 0..n {
            res_pc.push(res[j * d + i]);
            harm_pc.push(harm[j * d + i]);
        }
        cors_values.push(pearsonr(&res_pc, &harm_pc));
    }
    println!("Correlations (Rust vs R) per PC: {cors_values:?}");
    println!(
        "Min correlation: {:.3}",
        cors_values.iter().cloned().fold(f64::INFINITY, f64::min)
    );
    assert!(
        cors_values.iter().all(|&c| c >= min_correlation),
        "Some correlations < {min_correlation}: {cors_values:?}"
    );
}

/// Port of `test_run_harmony_small`.
#[test]
fn test_run_harmony_small() {
    run_harmony_golden(
        "pbmc_3500_meta.tsv.gz",
        "pbmc_3500_pcs.tsv.gz",
        "pbmc_3500_pcs_harmony2.tsv.gz",
        "donor",
        0.99,
    );
}

/// Port of `test_run_harmony_with_lab_and_day`: correcting for lab and day
/// should return finite coordinates for every cell.
#[test]
fn test_run_harmony_with_lab_and_day() {
    let n_cells = 120usize;
    let n_coordinates = 5usize;
    let random_seed = 0u32;
    let lab_labels = ["lab_a", "lab_b"];
    let day_labels = ["Monday", "Tuesday", "Wednesday"];

    // Deterministic repeatable data (xorshift for reproducibility).
    let mut state = random_seed as u64 | 1;
    let mut next_uniform = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut normal = || {
        // Box-Muller.
        let u1 = next_uniform().max(1e-12);
        let u2 = next_uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };

    let mut cell_coordinates = vec![0.0; n_cells * n_coordinates];
    for v in &mut cell_coordinates {
        *v = normal();
    }

    // Cell metadata: permuted tiled labels.
    let mut lab: Vec<String> = (0..n_cells)
        .map(|i| lab_labels[i % lab_labels.len()].to_string())
        .collect();
    let mut day: Vec<String> = (0..n_cells)
        .map(|i| day_labels[i % day_labels.len()].to_string())
        .collect();
    // Deterministic shuffle.
    for i in (1..n_cells).rev() {
        let j = (next_uniform() * (i + 1) as f64) as usize % (i + 1);
        lab.swap(i, j);
    }
    for i in (1..n_cells).rev() {
        let j = (next_uniform() * (i + 1) as f64) as usize % (i + 1);
        day.swap(i, j);
    }

    let mut cell_metadata: HashMap<String, Vec<String>> = HashMap::new();
    cell_metadata.insert("lab".to_string(), lab);
    cell_metadata.insert("day".to_string(), day);

    let result = run_harmony(&RunHarmonyArgs {
        max_iter_harmony: 2,
        max_iter_kmeans: 2,
        random_state: random_seed,
        ..RunHarmonyArgs::python_defaults(
            &cell_coordinates,
            n_cells,
            n_coordinates,
            &cell_metadata,
            vec!["lab".to_string(), "day".to_string()],
        )
    })
    .expect("run_harmony failed");

    assert_eq!(result.z_corr.len(), cell_coordinates.len());
    assert!(result.z_corr.iter().all(|v| v.is_finite()));
}

/// Port of `test_ridge_does_not_reverse_two_cells`.
#[test]
fn test_ridge_does_not_reverse_two_cells() {
    let values = [1.0, 2.0];
    let mut metadata: HashMap<String, Vec<String>> = HashMap::new();
    metadata.insert("lab".to_string(), vec!["A".into(), "B".into()]);
    metadata.insert("day".to_string(), vec!["Monday".into(), "Tuesday".into()]);

    let result = run_harmony(&RunHarmonyArgs {
        nclust: Some(1),
        max_iter_harmony: 1,
        max_iter_kmeans: 0,
        lamb: Some(ScalarOrVec::Scalar(0.5)),
        sigma: Some(ScalarOrVec::Vec(vec![1.0])),
        ..RunHarmonyArgs::python_defaults(
            &values,
            2,
            1,
            &metadata,
            vec!["lab".to_string(), "day".to_string()],
        )
    })
    .expect("run_harmony failed");

    // With equal weights and penalties, these values should move closer
    // without reversing their order. Z_corr is d x N column-major (d = 1).
    assert!(result.z_corr[0] < result.z_corr[1]);
}

/// Port of `optimizer_case`.
fn optimizer_case() -> (Vec<f64>, HashMap<String, Vec<String>>) {
    let data_mat = vec![
        1.0, 0.0, 0.8, 0.2, 0.0, 1.0, -1.0, 0.0, -0.8, -0.2, 0.0, -1.0, // row-major 6x2
    ];
    let mut meta_data: HashMap<String, Vec<String>> = HashMap::new();
    meta_data.insert(
        "batch".to_string(),
        vec!["a", "a", "b", "b", "a", "b"]
            .into_iter()
            .map(String::from)
            .collect(),
    );
    (data_mat, meta_data)
}

/// Port of `test_objective_convergence` (16 parametrized cases).
#[test]
fn test_objective_convergence() {
    let cases: &[(f64, f64, f64, bool)] = &[
        (100.0, 101.0, 0.01, false),
        (100.0, 99.5, 0.01, true),
        (100.0, 90.0, 0.01, false),
        (100.0, 99.0, 0.01, false),
        (-100.0, -100.5, 0.01, true),
        (-100.0, -99.0, 0.01, false),
        (0.0, 0.0, 0.01, true),
        (0.0, 1.0, 0.01, false),
        (0.0, -1.0, 0.01, false),
        (f64::NAN, 1.0, 0.01, false),
        (1.0, f64::NAN, 0.01, false),
        (f64::INFINITY, 1.0, 0.01, false),
        (1.0, f64::INFINITY, 0.01, false),
        (f64::NEG_INFINITY, 1.0, 0.01, false),
        (1.0, f64::NEG_INFINITY, 0.01, false),
        (1.0, 1.0, f64::NAN, false),
    ];
    for &(obj_old, obj_new, epsilon, expected) in cases {
        assert_eq!(
            objective_converged(obj_old as f32, obj_new as f32, epsilon as f32),
            expected,
            "case: {obj_old}, {obj_new}, {epsilon}"
        );
    }
}

/// Port of `test_objective_increase_does_not_stop_harmony`.
///
/// Verifies that harmony does *not* stop early when the objective rises:
/// it should still run all `max_iter_harmony` iterations. Under the ChaCha8Rng
/// backend the default seed's trajectory converges in one iteration, so we pin
/// `random_state: 42`, whose trajectory shows an objective increase between
/// iterations 0 and 1 yet still completes both iterations (len == 3).
#[test]
fn test_objective_increase_does_not_stop_harmony() {
    let (data_mat, meta_data) = optimizer_case();
    let ho = run_harmony(&RunHarmonyArgs {
        sigma: Some(ScalarOrVec::Scalar(1.0)),
        theta: Some(ScalarOrVec::Scalar(2.0)),
        nclust: Some(2),
        lamb: Some(ScalarOrVec::Scalar(1.0)),
        block_size: 0.5,
        max_iter_harmony: 2,
        max_iter_kmeans: 1,
        epsilon_cluster: 0.0,
        epsilon_harmony: 0.01,
        random_state: 42,
        ..RunHarmonyArgs::python_defaults(&data_mat, 6, 2, &meta_data, vec!["batch".to_string()])
    })
    .expect("run_harmony failed");

    assert!(ho.objective_harmony[1] > ho.objective_harmony[0]);
    assert_eq!(ho.objective_harmony.len(), 3);
}

/// Port of `assignment_inputs`.
#[allow(clippy::type_complexity)]
fn assignment_inputs() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<i64>) {
    let distances = vec![0.2, 1.1, 0.7, 1.2, 0.3, 0.6, 0.8, 0.9, 0.2]; // 3x3 row-major
    let sigma = vec![0.8, 1.2, 0.6];
    let e = vec![2.0, 4.0, 1.0, 3.0, 3.0, 2.0, 4.0, 1.0, 1.0, 3.0, 2.0, 4.0]; // 3x4
    let o = vec![3.0, 1.0, 2.0, 4.0, 1.0, 4.0, 3.0, 2.0, 2.0, 3.0, 1.0, 4.0]; // 3x4
    let theta = vec![2.0, 1.5, 0.5, 3.0];
    let batch_ids = vec![0, 1, 0, 2, 3, 3]; // 2x3 row-major
    (distances, sigma, e, o, theta, batch_ids)
}

/// Port of `old_assignment_probabilities` — the pre-harmony2 formula.
fn old_assignment_probabilities(
    distances: &[f64],
    sigma: &[f64],
    e: &[f64],
    o: &[f64],
    theta: &[f64],
    batch_ids: &[i64],
) -> Vec<f64> {
    let k = sigma.len();
    let n = distances.len() / k;
    // probabilities = exp(-distances / sigma[:, None])
    let mut probabilities = vec![0.0f64; k * n];
    for j in 0..n {
        for i in 0..k {
            probabilities[j * k + i] = (-distances[j * k + i] / sigma[i]).exp();
        }
    }
    // /= probabilities.sum(axis=0)
    for j in 0..n {
        let s: f64 = (0..k).map(|i| probabilities[j * k + i]).sum();
        for i in 0..k {
            probabilities[j * k + i] /= s;
        }
    }
    // ratio = (2E + 1) / (O + E + 1) — k x nb (rows are clusters).
    let (k, n) = (sigma.len(), distances.len() / sigma.len());
    let nb = theta.len();
    let ratio: Vec<f64> = (0..k * nb)
        .map(|idx| (2.0 * e[idx] + 1.0) / (o[idx] + e[idx] + 1.0))
        .collect();
    // for cell_batches in batch_ids: probabilities *= ratio[:, cell_batches]
    // ** theta[cell_batches] — elementwise over (cluster, cell); batch_ids is
    // n_covariates x n_cells, row-major.
    let n_cov = batch_ids.len() / n;
    for c in 0..n_cov {
        for j in 0..n {
            let b = batch_ids[c * n + j] as usize;
            for i in 0..k {
                probabilities[j * k + i] *= ratio[i * nb + b].powf(theta[b]);
            }
        }
    }
    // /= probabilities.sum(axis=0)
    for j in 0..n {
        let s: f64 = (0..k).map(|i| probabilities[j * k + i]).sum();
        for i in 0..k {
            probabilities[j * k + i] /= s;
        }
    }
    probabilities
}

/// Port of `assignment_probabilities` via the public internals: computes
/// logits, exponentiates shifted, normalizes (the same path the C++
/// `_assignment_probabilities` binding exposes).
fn assignment_probabilities(
    distances: &[f64],
    sigma: &[f64],
    e: &[f64],
    o: &[f64],
    theta: &[f64],
    batch_ids: &[i64],
) -> Vec<f64> {
    // Convert to f32 column-major.
    let (k, n) = (sigma.len(), distances.len() / sigma.len());
    let mut dist_mat = vec![0.0f32; k * n];
    for j in 0..n {
        for i in 0..k {
            dist_mat[j * k + i] = distances[j * k + i] as f32;
        }
    }
    let dist = harmony::fmat::FMat::from_col_major(k, n, dist_mat);
    let sig: Vec<f32> = sigma.iter().map(|&v| v as f32).collect();
    // E and O are k x nb, row-major.
    let nb = e.len() / k;
    let mut emat = vec![0.0f32; k * nb];
    for b in 0..nb {
        for i in 0..k {
            emat[b * k + i] = e[i * nb + b] as f32;
        }
    }
    let em = harmony::fmat::FMat::from_col_major(k, nb, emat);
    let mut omat = vec![0.0f32; k * nb];
    for b in 0..nb {
        for i in 0..k {
            omat[b * k + i] = o[i * nb + b] as f32;
        }
    }
    let om = harmony::fmat::FMat::from_col_major(k, nb, omat);
    let th: Vec<f32> = theta.iter().map(|&v| v as f32).collect();
    let ids: Vec<u32> = batch_ids.iter().map(|&v| v as u32).collect();

    let logits = harmony::harmony::core::assignment_logits(&dist, &sig, &em, &om, &th, &ids);
    let mut logits = logits;
    let normalizers = harmony::harmony::core::exponentiate_shifted_logits(&mut logits);
    logits.div_rows_by_col_vec(&normalizers);
    logits.data.iter().map(|&v| v as f64).collect()
}

/// Port of `test_log_assignments_match_moderate_multiplicative_weights`.
#[test]
fn test_log_assignments_match_moderate_multiplicative_weights() {
    let (distances, sigma, e, o, theta, batch_ids) = assignment_inputs();

    let actual = assignment_probabilities(&distances, &sigma, &e, &o, &theta, &batch_ids);
    let expected = old_assignment_probabilities(&distances, &sigma, &e, &o, &theta, &batch_ids);

    assert_allclose(&actual, &expected, 2e-6, 1e-7);
}

/// Port of `test_log_assignments_ignore_cell_logit_shifts`.
#[test]
fn test_log_assignments_ignore_cell_logit_shifts() {
    let (distances, sigma, e, o, theta, batch_ids) = assignment_inputs();
    let expected = assignment_probabilities(&distances, &sigma, &e, &o, &theta, &batch_ids);
    let k = sigma.len();
    let n = distances.len() / k;
    // shifted_distances = distances - sigma[:, None] * offsets
    let offsets = [30.0, -40.0, 20.0];
    let mut shifted = vec![0.0; k * n];
    for j in 0..n {
        for i in 0..k {
            shifted[j * k + i] = distances[j * k + i] - sigma[i] * offsets[j];
        }
    }
    let actual = assignment_probabilities(&shifted, &sigma, &e, &o, &theta, &batch_ids);
    assert_allclose(&actual, &expected, 2e-5, 1e-6);
}

/// Port of `test_log_assignments_preserve_covariate_penalties`.
#[test]
fn test_log_assignments_preserve_covariate_penalties() {
    let (distances, sigma, e, o, theta, batch_ids) = assignment_inputs();
    let mut neutral_theta = theta.clone();
    neutral_theta[2..].copy_from_slice(&vec![0.0; theta.len() - 2]);

    let one_covariate =
        assignment_probabilities(&distances, &sigma, &e, &o, &neutral_theta, &batch_ids[..3]);
    let neutral_covariate =
        assignment_probabilities(&distances, &sigma, &e, &o, &neutral_theta, &batch_ids);
    // batch_ids[::-1] in numpy reverses the rows (covariates).
    let reversed_ids = batch_ids.clone();
    let n_cov = reversed_ids.len() / 3;
    let mut flat_rev = Vec::with_capacity(reversed_ids.len());
    for c in (0..n_cov).rev() {
        flat_rev.extend_from_slice(&reversed_ids[c * 3..c * 3 + 3]);
    }
    let reversed_covariates =
        assignment_probabilities(&distances, &sigma, &e, &o, &theta, &flat_rev);
    let original = assignment_probabilities(&distances, &sigma, &e, &o, &theta, &batch_ids);

    // neutral_covariate vs one_covariate: atol=1e-7
    for (a, b) in neutral_covariate.iter().zip(one_covariate.iter()) {
        assert!((a - b).abs() <= 1e-7, "{a} vs {b}");
    }
    // reversed vs original: atol=1e-7
    for (a, b) in reversed_covariates.iter().zip(original.iter()) {
        assert!((a - b).abs() <= 1e-7, "{a} vs {b}");
    }
}

/// Port of `test_extreme_theta_and_unbalanced_batches_remain_finite`.
#[test]
fn test_extreme_theta_and_unbalanced_batches_remain_finite() {
    let distances = vec![0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.1, 0.9, 0.7, 0.5, 0.3, 0.1]; // 2x6
    let sigma = vec![0.1, 0.1];
    let e = vec![5.0, 1.0, 5.0, 1.0]; // 2x2
    let o = vec![5.0, 0.0, 0.0, 1.0]; // 2x2
    let theta = vec![1000.0, 1000.0];
    let batch_ids = vec![0, 0, 0, 0, 0, 1];

    let probabilities = assignment_probabilities(&distances, &sigma, &e, &o, &theta, &batch_ids);

    assert!(probabilities.iter().all(|v| v.is_finite()));
    assert!(probabilities.iter().all(|&v| v >= 0.0));
    // Sum over clusters (axis=0 per column j).
    let k = sigma.len();
    for j in 0..6 {
        let s: f64 = (0..k).map(|i| probabilities[j * k + i]).sum();
        assert!((s - 1.0).abs() <= 1e-7, "column {j} sums to {s}");
    }

    let (data_mat, _) = optimizer_case();
    let mut meta_data: HashMap<String, Vec<String>> = HashMap::new();
    meta_data.insert(
        "batch".to_string(),
        vec!["a", "a", "a", "a", "a", "b"]
            .into_iter()
            .map(String::from)
            .collect(),
    );
    let ho = run_harmony(&RunHarmonyArgs {
        sigma: Some(ScalarOrVec::Scalar(1e-4)),
        theta: Some(ScalarOrVec::Scalar(1000.0)),
        nclust: Some(2),
        lamb: Some(ScalarOrVec::Scalar(1.0)),
        block_size: 0.5,
        max_iter_harmony: 1,
        max_iter_kmeans: 1,
        ..RunHarmonyArgs::python_defaults(&data_mat, 6, 2, &meta_data, vec!["batch".to_string()])
    })
    .expect("run_harmony failed");

    assert!(ho.r.iter().all(|v| v.is_finite()));
    assert!(ho.objective_harmony.iter().all(|v| v.is_finite()));
    assert!(ho.z_corr.iter().all(|v| v.is_finite()));
    // R sums along axis=1 (clusters per cell): R is N x K, check rows.
    for j in 0..ho.n {
        let s: f64 = (0..ho.k).map(|i| ho.r[j * ho.k + i]).sum();
        assert!((s - 1.0).abs() <= 1e-7, "cell {j} R sums to {s}");
    }
}

/// Port of `test_small_sigma_assignments_remain_finite`.
#[test]
fn test_small_sigma_assignments_remain_finite() {
    let (data_mat, meta_data) = optimizer_case();

    let ho = run_harmony(&RunHarmonyArgs {
        sigma: Some(ScalarOrVec::Scalar(1e-4)),
        nclust: Some(2),
        lamb: Some(ScalarOrVec::Scalar(1.0)),
        block_size: 0.5,
        max_iter_harmony: 1,
        max_iter_kmeans: 1,
        ..RunHarmonyArgs::python_defaults(&data_mat, 6, 2, &meta_data, vec!["batch".to_string()])
    })
    .expect("run_harmony failed");

    assert!(ho.r.iter().all(|v| v.is_finite()));
    assert!(ho.objective_harmony.iter().all(|v| v.is_finite()));
    assert!(ho.z_corr.iter().all(|v| v.is_finite()));
    assert!(ho.r.iter().all(|&v| v >= 0.0));
    for j in 0..ho.n {
        let s: f64 = (0..ho.k).map(|i| ho.r[j * ho.k + i]).sum();
        assert!((s - 1.0).abs() <= 1e-7, "cell {j} R sums to {s}");
    }
}

/// Port of `test_nonfinite_assignments_raise_numerical_error`.
#[test]
fn test_nonfinite_assignments_raise_numerical_error() {
    let (data_mat, meta_data) = optimizer_case();

    let result = run_harmony(&RunHarmonyArgs {
        sigma: Some(ScalarOrVec::Scalar(f64::NAN)),
        nclust: Some(2),
        lamb: Some(ScalarOrVec::Scalar(1.0)),
        max_iter_harmony: 1,
        max_iter_kmeans: 1,
        ..RunHarmonyArgs::python_defaults(&data_mat, 6, 2, &meta_data, vec!["batch".to_string()])
    });

    let err: HarmonyError = result.expect_err("expected numerical error");
    let message = err.0;
    assert!(
        message.contains("assignment normalizers must be finite and positive"),
        "message: {message}"
    );
    assert!(message.contains("sigma=[nan, nan]"), "message: {message}");
    assert!(message.contains("theta=[2, 2]"), "message: {message}");
    assert!(message.contains("block_size=0.05"), "message: {message}");
}

/// Port of `test_random_seed`.
#[test]
fn test_random_seed() {
    let (_, meta_cols) = read_tsv(&data_path("pbmc_3500_meta.tsv.gz"));
    let (pcs_header, pcs_cols) = read_tsv(&data_path("pbmc_3500_pcs.tsv.gz"));
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

    let run = |random_state: u32| -> Vec<f64> {
        run_harmony(&RunHarmonyArgs {
            max_iter_harmony: 2,
            max_iter_kmeans: 2,
            random_state,
            ..RunHarmonyArgs::python_defaults(&data_mat, n, d, &meta_map, vec!["donor".to_string()])
        })
        .expect("run_harmony failed")
        .z_corr
    };

    // Same seed produces the same results.
    let result1 = run(42);
    let result2 = run(42);
    let diff_same_seed: f64 = result1
        .iter()
        .zip(result2.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    println!("Difference between two runs with same seed: {diff_same_seed:.6}");
    assert!(
        diff_same_seed == 0.0,
        "same seed must be exactly reproducible"
    );

    // Different seeds produce different results.
    let result3 = run(123);
    let result4 = run(456);
    let diff_diff_seed: f64 = result3
        .iter()
        .zip(result4.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    println!("Difference between runs with different seeds: {diff_diff_seed:.2}");
    assert!(
        diff_diff_seed > 1000.0,
        "Expected diff > 1000, got {diff_diff_seed}"
    );
}

/// Port of `np.testing.assert_allclose(actual, expected, rtol, atol)`.
fn assert_allclose(actual: &[f64], expected: &[f64], rtol: f64, atol: f64) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let tol = atol + rtol * e.abs();
        assert!(
            (a - e).abs() <= tol,
            "mismatch at {i}: actual {a}, expected {e} (tol {tol})"
        );
    }
}
