//! Port of the Python wrapper layer: `run_harmony()` in
//! `../harmonypy/harmonypy/harmony.py` and `compute_lisi()` in
//! `../harmonypy/harmonypy/lisi.py`.
//!
//! Argument normalization (theta/lamb/sigma expansion, nclust default,
//! batch-of-cell construction with `np.unique` sorted codes, orientation
//! handling, error messages) mirrors the Python code exactly.

use crate::harmony::core::{Harmony, HarmonyError};
use crate::lisi::LisiError;
use std::collections::HashMap;

/// Either a scalar or a vector — mirrors Python's `float or list` parameters.
#[derive(Debug, Clone)]
pub enum ScalarOrVec {
    Scalar(f64),
    Vec(Vec<f64>),
}

impl ScalarOrVec {
    #[allow(dead_code)]
    fn as_vec(&self) -> Vec<f64> {
        match self {
            ScalarOrVec::Scalar(v) => vec![*v],
            ScalarOrVec::Vec(v) => v.clone(),
        }
    }
}

/// Port of the `Harmony` result object in harmony.py.
///
/// Shapes match the Python-visible orientations: Z_corr/Z_orig/Z_cos are N×d,
/// R is N×K, Y is d×K. Values are f64 (converted from the f32 internal state
/// at the API boundary, as the nanobind bindings do).
#[derive(Debug, Clone)]
pub struct HarmonyResult {
    pub z_corr: Vec<f64>,
    pub z_orig: Vec<f64>,
    pub z_cos: Vec<f64>,
    pub r: Vec<f64>,
    pub y: Vec<f64>,
    pub k: usize,
    pub n: usize,
    pub d: usize,
    pub objective_harmony: Vec<f64>,
    pub objective_kmeans: Vec<f64>,
    /// Diagnostic components of `objective_kmeans` (not exposed by the Python
    /// bindings; used by the differential parity gates).
    pub objective_kmeans_dist: Vec<f64>,
    pub objective_kmeans_entropy: Vec<f64>,
    pub objective_kmeans_cross: Vec<f64>,
    pub kmeans_rounds: Vec<usize>,
}

/// Arguments for `run_harmony` — mirrors the Python keyword parameters.
/// Defaults noted in the doc comments match the Python defaults; construct
/// explicitly (there is no `Default` impl because borrowed fields cannot be
/// defaulted).
pub struct RunHarmonyArgs<'a> {
    /// PCA embedding matrix, row-major. Cells × PCs or PCs × cells (one
    /// dimension must match the number of cells; auto-transposed like Python).
    pub data_mat: &'a [f64],
    pub data_rows: usize,
    pub data_cols: usize,
    /// Metadata with batch variables: cells × variables.
    pub meta_data: &'a HashMap<String, Vec<String>>,
    /// Column name(s) in meta_data to use for batch correction.
    pub vars_use: Vec<String>,
    /// Diversity penalty parameter(s). Default: 2 for each batch (None).
    pub theta: Option<ScalarOrVec>,
    /// Ridge regression penalty. Default: None (auto-estimation).
    pub lamb: Option<ScalarOrVec>,
    /// Kernel bandwidth for soft clustering. Default: 0.1.
    pub sigma: Option<ScalarOrVec>,
    /// Number of clusters. Default: min(N/30, 100).
    pub nclust: Option<usize>,
    /// Protection against overcorrection. Default: 0.
    pub tau: f64,
    /// Proportion of cells to update in each block. Default: 0.05.
    pub block_size: f64,
    /// Maximum Harmony iterations. Default: 10.
    pub max_iter_harmony: usize,
    /// Maximum k-means iterations per Harmony iteration. Default: 4.
    pub max_iter_kmeans: usize,
    /// K-means convergence threshold. Default: 1e-3.
    pub epsilon_cluster: f64,
    /// Harmony convergence threshold. Default: 1e-2.
    pub epsilon_harmony: f64,
    /// Alpha parameter for lambda estimation. Default: 0.2.
    pub alpha: f64,
    /// Minimum batch proportion in a cluster for correction. Default: 1e-5.
    pub batch_prop_cutoff: f64,
    /// Random seed for reproducibility. Default: 0.
    pub random_state: u32,
}

/// Python-default argument values, for callers that want them.
impl<'a> RunHarmonyArgs<'a> {
    pub fn python_defaults(
        data_mat: &'a [f64],
        data_rows: usize,
        data_cols: usize,
        meta_data: &'a HashMap<String, Vec<String>>,
        vars_use: Vec<String>,
    ) -> Self {
        RunHarmonyArgs {
            data_mat,
            data_rows,
            data_cols,
            meta_data,
            vars_use,
            theta: None,
            lamb: None,
            sigma: None,
            nclust: None,
            tau: 0.0,
            block_size: 0.05,
            max_iter_harmony: 10,
            max_iter_kmeans: 4,
            epsilon_cluster: 1e-3,
            epsilon_harmony: 1e-2,
            alpha: 0.2,
            batch_prop_cutoff: 1e-5,
            random_state: 0,
        }
    }
}

/// Port of `run_harmony()` in harmony.py.
pub fn run_harmony(args: &RunHarmonyArgs<'_>) -> Result<HarmonyResult, HarmonyError> {
    let mut data_mat = args.data_mat.to_vec();
    let mut data_rows = args.data_rows;
    let mut data_cols = args.data_cols;

    let nclust_in = args.nclust;
    let sigma_in = args.sigma.clone();
    let theta_in = args.theta.clone();
    let lamb_in = args.lamb.clone();

    // N from the metadata (mirrors meta_data.shape[0] / len(meta_data[first]).
    let n = args
        .meta_data
        .get(&args.vars_use[0])
        .map(|v| v.len())
        .unwrap_or(0);

    // Orientation handling: if data shape[1] == N pass; elif shape[0] == N
    // transpose; else error (message mirrors the Python ValueError).
    if data_cols == n {
        // pass
    } else if data_rows == n {
        // transpose (row-major)
        let mut transposed = vec![0.0; data_rows * data_cols];
        for i in 0..data_rows {
            for j in 0..data_cols {
                transposed[j * data_rows + i] = data_mat[i * data_cols + j];
            }
        }
        data_mat = transposed;
        std::mem::swap(&mut data_rows, &mut data_cols);
    } else {
        return Err(HarmonyError(format!(
            "data_mat has shape ({}, {}), but meta_data has {} cells. One dimension of data_mat must match the number of cells.",
            data_rows, data_cols, n
        )));
    }

    let d = data_rows;

    // nclust default: int(min(round(N / 30.0), 100)).
    let nclust = nclust_in.unwrap_or_else(|| ((n as f64 / 30.0).round().min(100.0)) as usize);

    // sigma handling: scalar float repeated if nclust > 1 (array passes
    // through). A scalar with nclust == 1 becomes a single-element vector.
    let sigma_vec: Vec<f64> = match &sigma_in {
        None => {
            if nclust > 1 {
                vec![0.1; nclust]
            } else {
                vec![0.1]
            }
        }
        Some(ScalarOrVec::Scalar(s)) => {
            if nclust > 1 {
                vec![*s; nclust]
            } else {
                vec![*s]
            }
        }
        Some(ScalarOrVec::Vec(v)) => v.clone(),
    };

    // Build compact batch-of-cell index (n_covariates x N), using sorted
    // unique values (np.unique semantics: codes follow sorted order).
    let ncov = args.vars_use.len();
    let mut batch_of_cell = vec![0i64; ncov * n];
    let mut phi_n = vec![0usize; ncov];
    let mut offset = 0usize;
    for (c, var) in args.vars_use.iter().enumerate() {
        let col = args
            .meta_data
            .get(var)
            .ok_or_else(|| HarmonyError(format!("metadata column '{var}' not found")))?;
        let (uniques, codes) = sorted_unique(col);
        phi_n[c] = uniques.len();
        for (j, &code) in codes.iter().enumerate() {
            batch_of_cell[c * n + j] = (code + offset) as i64;
        }
        offset += uniques.len();
    }
    let b_total = offset; // B = sum(phi_n)

    // Theta handling — default is 2 (matches R package).
    let theta_vec: Vec<f64> = match theta_in {
        None => repeat_by_var(2.0, &phi_n),
        Some(ScalarOrVec::Scalar(t)) => repeat_by_var(t, &phi_n),
        Some(ScalarOrVec::Vec(ref t)) if t.len() == ncov => repeat_by_var_from_slice(t, &phi_n),
        Some(ScalarOrVec::Vec(ref t)) => t.clone(),
    };
    if theta_vec.len() != b_total {
        return Err(HarmonyError(
            "each batch variable must have a theta".to_string(),
        ));
    }

    // Lambda handling (matches R harmony2: None = auto-estimation). Lambda
    // estimation is signalled to the core with the sentinel value -1
    // (mirrors harmony.py: "Signal lambda estimation with sentinel [-1]").
    let lambda_estimation;
    let lamb_vec: Vec<f64>;
    match &lamb_in {
        None => {
            lambda_estimation = true;
            lamb_vec = vec![-1.0];
        }
        Some(ScalarOrVec::Scalar(l)) if *l == -1.0 => {
            lambda_estimation = true;
            lamb_vec = vec![-1.0];
        }
        Some(ScalarOrVec::Scalar(l)) => {
            lambda_estimation = false;
            let mut v = repeat_by_var(*l, &phi_n);
            v.insert(0, 0.0);
            lamb_vec = v;
        }
        Some(ScalarOrVec::Vec(l)) if l.len() == ncov => {
            lambda_estimation = false;
            let mut v = repeat_by_var_from_slice(l, &phi_n);
            v.insert(0, 0.0);
            lamb_vec = v;
        }
        Some(ScalarOrVec::Vec(l)) => {
            if l.len() == b_total {
                lambda_estimation = false;
                let mut v = l.clone();
                v.insert(0, 0.0);
                lamb_vec = v;
            } else {
                // Full vector with intercept already present.
                lambda_estimation = false;
                lamb_vec = l.clone();
            }
        }
    }

    // N_b = bincount(batch_of_cell.ravel(), minlength=B); Pr_b = N_b / N.
    let mut n_b = vec![0.0f32; b_total];
    for &code in &batch_of_cell {
        n_b[code as usize] += 1.0;
    }
    let pr_b: Vec<f64> = n_b.iter().map(|&v| (v as f64) / n as f64).collect();

    // tau > 0: theta *= (1 - exp(-(N_b / (nclust * tau))^2)).
    let mut theta_vec = theta_vec;
    if args.tau > 0.0 {
        let nclust_tau = nclust as f64 * args.tau;
        for (bidx, t) in theta_vec.iter_mut().enumerate() {
            let ratio = n_b[bidx] as f64 / nclust_tau;
            *t *= 1.0 - (-(ratio * ratio)).exp();
        }
    }

    // Log parameters (content mirrors the Python layer).
    log::info!("Running Harmony");
    log::info!("  Parameters:");
    log::info!("    max_iter_harmony: {}", args.max_iter_harmony);
    log::info!("    max_iter_kmeans: {}", args.max_iter_kmeans);
    log::info!("    epsilon_cluster: {}", args.epsilon_cluster);
    log::info!("    epsilon_harmony: {}", args.epsilon_harmony);
    log::info!("    nclust: {}", nclust);
    log::info!("    block_size: {}", args.block_size);
    if lambda_estimation {
        log::info!("    lamb: dynamic (alpha={})", args.alpha);
    } else {
        log::info!("    lamb: {:?}", &lamb_vec[1..]);
    }
    log::info!("    theta: {:?}", theta_vec);
    log::info!("    verbose: true");
    log::info!("    random_state: {}", args.random_state);
    log::info!("  Data: {} PCs × {} cells", d, n);
    log::info!("  Batch variables: {:?}", args.vars_use);

    let harmony = Harmony::new(
        &data_mat,
        d,
        n,
        &batch_of_cell,
        &pr_b,
        &sigma_vec,
        &theta_vec,
        &lamb_vec,
        args.alpha,
        args.max_iter_harmony,
        args.max_iter_kmeans,
        args.epsilon_cluster,
        args.epsilon_harmony,
        nclust,
        args.block_size,
        &phi_n,
        args.batch_prop_cutoff,
        args.random_state,
    )?;

    Ok(HarmonyResult {
        z_corr: harmony.z_corr.data.iter().map(|&v| v as f64).collect(),
        z_orig: harmony.z_orig.data.iter().map(|&v| v as f64).collect(),
        z_cos: harmony.z_corr.data.iter().map(|&v| v as f64).collect(),
        r: harmony.r.data.iter().map(|&v| v as f64).collect(),
        y: harmony.y.data.iter().map(|&v| v as f64).collect(),
        k: harmony.k,
        n: harmony.n,
        d: harmony.d,
        objective_harmony: harmony
            .objective_harmony
            .iter()
            .map(|&v| v as f64)
            .collect(),
        objective_kmeans: harmony.objective_kmeans.iter().map(|&v| v as f64).collect(),
        objective_kmeans_dist: harmony
            .objective_kmeans_dist
            .iter()
            .map(|&v| v as f64)
            .collect(),
        objective_kmeans_entropy: harmony
            .objective_kmeans_entropy
            .iter()
            .map(|&v| v as f64)
            .collect(),
        objective_kmeans_cross: harmony
            .objective_kmeans_cross
            .iter()
            .map(|&v| v as f64)
            .collect(),
        kmeans_rounds: harmony.kmeans_rounds.clone(),
    })
}

/// Port of `compute_lisi()` in lisi.py.
///
/// `x` is N x d, row-major; `metadata` maps label column names to per-cell
/// string labels.
pub fn compute_lisi(
    x: &[f64],
    n_cells: usize,
    n_dims: usize,
    metadata: &HashMap<String, Vec<String>>,
    label_colnames: &[String],
    perplexity: f64,
) -> Result<Vec<f64>, LisiError> {
    let n_labels = label_colnames.len();
    let mut lisi_df = vec![0.0f64; n_cells * n_labels];
    for (i, label) in label_colnames.iter().enumerate() {
        let col = metadata
            .get(label)
            .ok_or_else(|| LisiError(format!("metadata column '{label}' not found")))?;
        let (uniques, codes) = sorted_unique(col);
        let n_categories = uniques.len();
        let values =
            crate::lisi::compute_lisi_impl(x, n_cells, n_dims, &codes, n_categories, perplexity);
        for j in 0..n_cells {
            lisi_df[j * n_labels + i] = values[j];
        }
    }
    Ok(lisi_df)
}

/// Port of `np.unique(values, return_inverse=True)` — sorted unique values
/// and inverse codes.
pub fn sorted_unique(values: &[String]) -> (Vec<String>, Vec<usize>) {
    let mut sorted: Vec<&String> = values.iter().collect();
    sorted.sort();
    sorted.dedup();
    let uniques: Vec<String> = sorted.into_iter().cloned().collect();
    let codes: Vec<usize> = values
        .iter()
        .map(|v| uniques.iter().position(|u| u == v).unwrap())
        .collect();
    (uniques, codes)
}

/// Port of `np.repeat([t] * len(phi_n), phi_n)` — each variable's value
/// repeated across its levels.
fn repeat_by_var(t: f64, phi_n: &[usize]) -> Vec<f64> {
    let mut out = Vec::new();
    for &count in phi_n {
        for _ in 0..count {
            out.push(t);
        }
    }
    out
}

fn repeat_by_var_from_slice(values: &[f64], phi_n: &[usize]) -> Vec<f64> {
    let mut out = Vec::new();
    for (&v, &count) in values.iter().zip(phi_n.iter()) {
        for _ in 0..count {
            out.push(v);
        }
    }
    out
}
