//! The Harmony batch correction core (itself a port of the R harmony2
//! package, using scatter/gather kernels on a batch_id vector instead of
//! sparse Phi matrices).
//!
//! All state is f32 (MATTYPE = arma::fmat), matching the C++ backend.
//!
//! Index-based loops are intentional throughout: they mirror the reference's
//! flat-buffer (column-major) indexing directly.

#![allow(clippy::needless_range_loop)]

use crate::fmat::FMat;
use crate::harmony::kmeans::kmeans_init;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::fmt;

pub mod cpp_ostream {
    /// Port of `std::ostream << float` (libstdc++ default formatting:
    /// `%g` with precision 6, e.g. `nan`, `inf`, `-inf`, `2`, `0.05`,
    /// `1e-05`, `1.23457e+07`).
    pub fn float(v: f32) -> String {
        if v.is_nan() {
            return "nan".to_string();
        }
        if v.is_infinite() {
            return if v < 0.0 {
                "-inf".to_string()
            } else {
                "inf".to_string()
            };
        }
        let f = v as f64;
        if f == 0.0 {
            return if f.is_sign_negative() { "-0" } else { "0" }.to_string();
        }
        let exp = f.abs().log10().floor() as i32;
        if !(-4..6).contains(&exp) {
            // Scientific: %g with precision 5 fraction digits, trailing
            // zeros stripped, exponent at least two digits with sign.
            let mantissa = f / (exp as f64).exp2().powf(std::f64::consts::LN_10); // 10^exp
            let mantissa = format!("{:.*}", 5, mantissa);
            let mantissa = mantissa
                .trim_end_matches('0')
                .trim_end_matches('.')
                .to_string();
            format!(
                "{mantissa}e{}{:02}",
                if exp < 0 { "-" } else { "+" },
                exp.abs()
            )
        } else {
            // Fixed: precision 6 - 1 - exp, trailing zeros stripped.
            let decimals = (5 - exp).max(0) as usize;
            let s = format!("{:.*}", decimals, f);
            let s = if s.contains('.') {
                s.trim_end_matches('0').trim_end_matches('.')
            } else {
                &s
            };
            s.to_string()
        }
    }
}

/// Error type mirroring the C++ `std::runtime_error` messages.
#[derive(Debug, Clone)]
pub struct HarmonyError(pub String);

impl fmt::Display for HarmonyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for HarmonyError {}

pub struct Harmony {
    pub z_orig: FMat, // d x N
    pub z_corr: FMat, // d x N

    /// n_covariates x N: batch index per covariate (row-major).
    pub batch_ids: Vec<u32>,
    pub n_covariates: usize,
    pub pr_b: Vec<f32>,        // B
    pub batch_sizes: Vec<f32>, // B
    /// Per-batch cell index lists (port of `std::vector<arma::uvec> batch_index`).
    pub batch_index: Vec<Vec<u32>>,

    pub y: FMat,        // d x K
    pub r: FMat,        // K x N
    pub dist_mat: FMat, // K x N

    pub o: FMat, // K x B
    pub e: FMat, // K x B
    pub w: FMat, // (B+1) x d

    pub sigma: Vec<f32>,
    pub theta: Vec<f32>,
    pub lambda: Vec<f32>,

    pub alpha: f32,
    pub lambda_estimation: bool,

    pub n: usize,
    pub d: usize,
    pub k: usize,
    pub b: usize,
    pub max_iter_harmony: usize,
    pub max_iter_kmeans: usize,
    pub epsilon_kmeans: f32,
    pub epsilon_harmony: f32,
    pub block_size: f32,
    pub window_size: usize,

    pub b_vec: Vec<usize>,
    pub covariate_bounds: Vec<usize>,
    pub batch_proportion_cutoff: f32,

    pub objective_harmony: Vec<f32>,
    pub objective_kmeans: Vec<f32>,
    pub objective_kmeans_dist: Vec<f32>,
    pub objective_kmeans_entropy: Vec<f32>,
    pub objective_kmeans_cross: Vec<f32>,
    pub kmeans_rounds: Vec<usize>,

    pub rng: ChaCha8Rng,
}

impl Harmony {
    /// Port of `Harmony::Harmony(...)` in harmony.cpp. Runs initialization and
    /// harmonization on construction, mirroring the C++ constructor.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        z: &[f64],             // row-major, (d, N) — converted f64 → f32
        n_rows: usize,         // d
        n_cols: usize,         // N
        batch_of_cell: &[i64], // row-major (n_cov, N)
        pr_b_in: &[f64],
        sigma_in: &[f64],
        theta_in: &[f64],
        lambda_in: &[f64],
        alpha_in: f64,
        max_iter_harmony: usize,
        max_iter_kmeans: usize,
        epsilon_kmeans: f64,
        epsilon_harmony: f64,
        k: usize,
        block_size: f64,
        b_vec_in: &[usize],
        batch_proportion_cutoff: f64,
        random_state: u32,
    ) -> Result<Harmony, HarmonyError> {
        let d = n_rows;
        let n = n_cols;

        let mut h = Harmony {
            z_orig: FMat::from_row_major_f64(d, n, z),
            z_corr: FMat::zeros(d, n),
            batch_ids: Vec::new(),
            n_covariates: 0,
            pr_b: Vec::new(),
            batch_sizes: Vec::new(),
            batch_index: Vec::new(),
            y: FMat::zeros(d, k),
            r: FMat::zeros(k, n),
            dist_mat: FMat::zeros(k, n),
            o: FMat::zeros(k, 0),
            e: FMat::zeros(k, 0),
            w: FMat::zeros(0, d),
            sigma: sigma_in.iter().map(|&v| v as f32).collect(),
            theta: theta_in.iter().map(|&v| v as f32).collect(),
            lambda: Vec::new(),
            alpha: alpha_in as f32,
            lambda_estimation: false,
            n,
            d,
            k,
            b: b_vec_in.iter().sum(),
            max_iter_harmony,
            max_iter_kmeans,
            epsilon_kmeans: epsilon_kmeans as f32,
            epsilon_harmony: epsilon_harmony as f32,
            block_size: block_size as f32,
            window_size: 3,
            b_vec: b_vec_in.to_vec(),
            covariate_bounds: Vec::new(),
            batch_proportion_cutoff: batch_proportion_cutoff as f32,
            objective_harmony: Vec::new(),
            objective_kmeans: Vec::new(),
            objective_kmeans_dist: Vec::new(),
            objective_kmeans_entropy: Vec::new(),
            objective_kmeans_cross: Vec::new(),
            kmeans_rounds: Vec::new(),
            rng: ChaCha8Rng::seed_from_u64(random_state as u64),
        };

        h.z_corr = h.z_orig.normalise_cols2();

        h.pr_b = pr_b_in.iter().map(|&v| v as f32).collect();

        if lambda_in[0] < 0.0 {
            h.lambda_estimation = true;
            h.lambda = vec![0.0; h.b + 1];
        } else {
            h.lambda_estimation = false;
            h.lambda = lambda_in.iter().map(|&v| v as f32).collect();
        }

        if h.b_vec.len() > 1 {
            // std::partial_sum
            let mut acc = 0usize;
            for &v in &h.b_vec {
                acc += v;
                h.covariate_bounds.push(acc);
            }
        } else {
            h.covariate_bounds.push(h.b_vec[0]);
        }

        h.build_batch_structures(batch_of_cell);
        h.allocate_buffers();

        log::info!("Computing initial centroids...");
        h.init_cluster()?;
        h.check_state("initialization")?;
        log::info!("Initialization complete.");
        h.harmonize(max_iter_harmony)?;
        h.check_state("return")?;
        Ok(h)
    }

    /// Port of `Harmony::build_batch_structures`.
    fn build_batch_structures(&mut self, batch_of_cell: &[i64]) {
        let ncov = self.b_vec.len();
        self.n_covariates = ncov;
        self.batch_ids = vec![0u32; ncov * self.n];
        for c in 0..ncov {
            for j in 0..self.n {
                self.batch_ids[c * self.n + j] = batch_of_cell[c * self.n + j] as u32;
            }
        }

        // Compute batch sizes (count cells per batch across all covariates).
        self.batch_sizes = vec![0.0f32; self.b];
        for c in 0..ncov {
            for j in 0..self.n {
                self.batch_sizes[self.batch_ids[c * self.n + j] as usize] += 1.0;
            }
        }
        // Each cell is counted n_covariates times; normalize.
        let ncov_f = ncov as f32;
        for v in &mut self.batch_sizes {
            *v /= ncov_f;
        }

        // Determine which covariate owns each batch.
        let mut cov_of_batch = vec![0usize; self.b];
        let mut idx = 0usize;
        for (c, &cnt) in self.b_vec.iter().enumerate() {
            for _ in 0..cnt {
                cov_of_batch[idx] = c;
                idx += 1;
            }
        }

        // Count cells per batch (using the owning covariate).
        let mut counts = vec![0usize; self.b];
        for b in 0..self.b {
            let c = cov_of_batch[b];
            for j in 0..self.n {
                if self.batch_ids[c * self.n + j] as usize == b {
                    counts[b] += 1;
                }
            }
        }

        // Fill batch_index.
        self.batch_index = counts.iter().map(|&c| Vec::with_capacity(c)).collect();
        for b in 0..self.b {
            let c = cov_of_batch[b];
            for j in 0..self.n {
                if self.batch_ids[c * self.n + j] as usize == b {
                    self.batch_index[b].push(j as u32);
                }
            }
        }
    }

    /// Port of `Harmony::allocate_buffers`.
    fn allocate_buffers(&mut self) {
        self.dist_mat = FMat::zeros(self.k, self.n);
        self.o = FMat::zeros(self.k, self.b);
        self.e = FMat::zeros(self.k, self.b);
        self.w = FMat::zeros(self.b + 1, self.d);
        self.r = FMat::zeros(self.k, self.n);
        self.y = FMat::zeros(self.d, self.k);
    }

    /// Port of `Harmony::numerical_error` — builds the exact C++ message.
    fn numerical_error(&self, stage: &str, invariant: &str) -> HarmonyError {
        let mut msg = format!("Harmony numerical error during {stage}: {invariant} (sigma=[");
        for (i, s) in self.sigma.iter().enumerate() {
            if i > 0 {
                msg.push_str(", ");
            }
            msg.push_str(&cpp_ostream::float(*s));
        }
        msg.push_str("], theta=[");
        for (i, t) in self.theta.iter().enumerate() {
            if i > 0 {
                msg.push_str(", ");
            }
            msg.push_str(&cpp_ostream::float(*t));
        }
        msg.push_str(&format!(
            "], block_size={}, K={}, N={})",
            cpp_ostream::float(self.block_size),
            self.k,
            self.n
        ));
        HarmonyError(msg)
    }

    /// Port of `Harmony::normalize_log_assignments`. Returns the normalizers
    /// on success (associated fn to avoid borrow conflicts; call sites
    /// construct the numerical error via `numerical_error`).
    fn normalize_log_assignments(logits: &mut FMat) -> Option<Vec<f32>> {
        let normalizers = exponentiate_shifted_logits(logits);
        let finite = normalizers.iter().all(|v| v.is_finite());
        if !finite || normalizers.iter().any(|&v| v <= 0.0) {
            return None;
        }
        logits.div_rows_by_col_vec(&normalizers);
        Some(normalizers)
    }

    /// Port of `Harmony::check_state`.
    fn check_state(&self, stage: &str) -> Result<(), HarmonyError> {
        if !self.r.is_finite() {
            return Err(self.numerical_error(stage, "assignments must be finite"));
        }
        if self.r.min() < 0.0 {
            return Err(self.numerical_error(stage, "assignments must be nonnegative"));
        }

        let assignment_sums = self.r.row_sums();
        if assignment_sums.iter().any(|v| !v.is_finite())
            || assignment_sums.iter().any(|&v| (v - 1.0).abs() > 1e-4)
        {
            return Err(self.numerical_error(stage, "assignment columns must sum to one"));
        }

        let all_finite = |values: &[f32]| values.iter().all(|v| v.is_finite());
        if !all_finite(&self.objective_harmony)
            || !all_finite(&self.objective_kmeans)
            || !all_finite(&self.objective_kmeans_dist)
            || !all_finite(&self.objective_kmeans_entropy)
            || !all_finite(&self.objective_kmeans_cross)
        {
            return Err(self.numerical_error(stage, "objectives must be finite"));
        }

        if !self.z_corr.is_finite() {
            return Err(self.numerical_error(stage, "corrected coordinates must be finite"));
        }
        Ok(())
    }

    /// Port of `Harmony::scatter_add_O` — scatter-add R columns into O for
    /// each covariate's batch assignment. `ids` is n_cov x N (row-major).
    /// (Free function to avoid borrow conflicts.)
    fn scatter_add_o(o: &mut FMat, rsub: &FMat, ids: &[u32], n_covariates: usize, sign: f32) {
        let n = rsub.ncols;
        let k = rsub.nrows;
        for c in 0..n_covariates {
            for j in 0..n {
                let bidx = ids[c * n + j] as usize;
                let col = rsub.col(j);
                let dst = o.col_mut(bidx);
                if sign > 0.0 {
                    for i in 0..k {
                        dst[i] += col[i];
                    }
                } else {
                    for i in 0..k {
                        dst[i] -= col[i];
                    }
                }
            }
        }
    }

    /// Port of `Harmony::init_cluster`.
    fn init_cluster(&mut self) -> Result<(), HarmonyError> {
        self.y = kmeans_init(&self.z_corr, self.k, &mut self.rng);
        self.y = self.y.normalise_cols2();

        self.dist_mat = {
            let mut yt_z = self.y.mul_trans_a(&self.z_corr); // K x N
            for v in yt_z.data.iter_mut() {
                *v = 2.0 * (1.0 - *v);
            }
            yt_z
        };

        self.r = self.dist_mat.clone();
        for v in self.r.data.iter_mut() {
            *v = -*v;
        }
        self.r.div_each_col_by_row_vec(&self.sigma);
        if Self::normalize_log_assignments(&mut self.r).is_none() {
            return Err(self.numerical_error(
                "initialization",
                "assignment normalizers must be finite and positive",
            ));
        }

        let r_col_sums = self.r.col_sums(); // K
        self.e = FMat::zeros(self.k, self.b);
        for g in 0..self.k {
            for bidx in 0..self.b {
                self.e.set(g, bidx, r_col_sums[g] * self.pr_b[bidx]);
            }
        }
        self.o = FMat::zeros(self.k, self.b);
        {
            let ids = self.batch_ids.clone();
            Self::scatter_add_o(&mut self.o, &self.r, &ids, self.n_covariates, 1.0);
        }

        self.compute_objective();
        let last = *self.objective_kmeans.last().unwrap();
        self.objective_harmony.push(last);
        Ok(())
    }

    /// Port of `Harmony::compute_objective`.
    fn compute_objective(&mut self) {
        let norm_const = 2000.0f32 / self.n as f32;

        let kmeans_error = self.r.accu_product(&self.dist_mat);

        // log_R = R; transform(v > 0 ? v * log(v) : 0)
        let mut log_r = self.r.clone();
        for v in log_r.data.iter_mut() {
            *v = if *v > 0.0 { *v * v.ln() } else { 0.0 };
        }
        // _entropy = accu(log_R.each_col() % sigma)
        // (accu over an expression falls back to accu_proxy_linear's
        // dual-accumulator order over the column-major element sequence.)
        let mut val1 = 0.0f32;
        let mut val2 = 0.0f32;
        let mut idx = 0usize;
        for j in 0..log_r.ncols {
            for (i, &v) in log_r.col(j).iter().enumerate() {
                let p = v * self.sigma[i];
                if idx % 2 == 0 {
                    val1 += p;
                } else {
                    val2 += p;
                }
                idx += 1;
            }
        }
        let entropy = val1 + val2;

        // ratio = (O + E + 1) / (2E + 1); log; *= theta (rows); *= sigma (cols); *= O
        let mut ratio = FMat::zeros(self.k, self.b);
        for j in 0..self.b {
            for i in 0..self.k {
                let num = self.o.get(i, j) + self.e.get(i, j) + 1.0;
                let den = 2.0 * self.e.get(i, j) + 1.0;
                ratio.set(i, j, num / den);
            }
        }
        for v in ratio.data.iter_mut() {
            *v = v.ln();
        }
        ratio.mul_each_row_by_col_vec(&self.theta);
        ratio.mul_each_col_by_row_vec(&self.sigma);
        for j in 0..self.b {
            for i in 0..self.k {
                let v = ratio.get(i, j) * self.o.get(i, j);
                ratio.set(i, j, v);
            }
        }
        let cross_entropy = ratio.accu();
        self.objective_kmeans
            .push((kmeans_error + entropy + cross_entropy) * norm_const);
        self.objective_kmeans_dist.push(kmeans_error * norm_const);
        self.objective_kmeans_entropy.push(entropy * norm_const);
        self.objective_kmeans_cross.push(cross_entropy * norm_const);
    }

    /// Port of `Harmony::harmonize`.
    fn harmonize(&mut self, iter_harmony: usize) -> Result<(), HarmonyError> {
        let mut converged = false;
        for i in 1..=iter_harmony {
            log::info!("Iteration {i} of {iter_harmony}");
            self.cluster()?;
            self.moe_correct_ridge()?;
            converged = self.check_convergence(1);
            if converged {
                let plural = if i > 1 { "s" } else { "" };
                log::info!("Converged after {i} iteration{plural}");
                break;
            }
        }
        if !converged {
            log::info!("Stopped before convergence");
        }
        Ok(())
    }

    /// Port of `Harmony::cluster`.
    fn cluster(&mut self) -> Result<(), HarmonyError> {
        if self.objective_harmony.len() > 1 {
            self.z_corr = self.z_corr.normalise_cols2();
            self.dist_mat = {
                let mut yt_z = self.y.mul_trans_a(&self.z_corr); // K x N
                for v in yt_z.data.iter_mut() {
                    *v = 2.0 * (1.0 - *v);
                }
                yt_z
            };
            self.r = self.dist_mat.clone();
            for v in self.r.data.iter_mut() {
                *v = -*v;
            }
            self.r.div_each_col_by_row_vec(&self.sigma);
            if Self::normalize_log_assignments(&mut self.r).is_none() {
                return Err(self.numerical_error(
                    "cluster initialization",
                    "assignment normalizers must be finite and positive",
                ));
            }
            let r_col_sums = self.r.col_sums();
            for g in 0..self.k {
                for bidx in 0..self.b {
                    self.e.set(g, bidx, r_col_sums[g] * self.pr_b[bidx]);
                }
            }
            self.o = FMat::zeros(self.k, self.b);
            {
                let ids = self.batch_ids.clone();
                Self::scatter_add_o(&mut self.o, &self.r, &ids, self.n_covariates, 1.0);
            }
        }

        let mut rounds = 0usize;
        for i in 0..self.max_iter_kmeans {
            self.update_r()?;
            self.compute_objective();
            self.check_state("assignment update")?;

            if i > self.window_size && self.check_convergence(0) {
                rounds = i + 1;
                break;
            }
            rounds = i + 1;
        }

        self.kmeans_rounds.push(rounds);
        let last = *self.objective_kmeans.last().unwrap();
        self.objective_harmony.push(last);
        Ok(())
    }

    /// Port of `Harmony::update_R`.
    fn update_r(&mut self) -> Result<(), HarmonyError> {
        let n = self.n;
        let mut indices_vec: Vec<usize> = (0..n).collect();
        indices_vec.shuffle(&mut self.rng);

        // reverse_index[update_order[i]] = i
        let mut reverse_index = vec![0usize; n];
        for (i, &u) in indices_vec.iter().enumerate() {
            reverse_index[u] = i;
        }

        let n_blocks = (1.0f32 / self.block_size).ceil() as usize;
        let cells_per_block = ((n as f32 * self.block_size) as usize).max(1);

        // R = R.cols(update_order); dist_mat = dist_mat.cols(update_order)
        // (reordered in place, like the C++ — block writes below target the
        // reordered positions, restored by reverse_index at the end).
        self.r = self.r.cols(&indices_vec);
        self.dist_mat = self.dist_mat.cols(&indices_vec);
        // Shuffle batch_ids (n_cov x N) by column order.
        let ncov = self.n_covariates;
        let mut batch_ids_shuf = vec![0u32; ncov * n];
        for c in 0..ncov {
            for (j, &u) in indices_vec.iter().enumerate() {
                batch_ids_shuf[c * n + j] = self.batch_ids[c * n + u];
            }
        }

        let k = self.k;
        for i in 0..n_blocks {
            let idx_min = i * cells_per_block;
            let idx_max = ((i + 1) * cells_per_block) - 1;
            let idx_max = if i == n_blocks - 1 { n - 1 } else { idx_max };
            if idx_min >= n {
                break;
            }
            let block_len = idx_max - idx_min + 1;

            // E -= sum(Rcells, 1) * Pr_b.t()
            {
                let sums = self.r.cols_range_sums(idx_min, idx_max + 1); // K
                for g in 0..k {
                    for bidx in 0..self.b {
                        let v = self.e.get(g, bidx) - sums[g] * self.pr_b[bidx];
                        self.e.set(g, bidx, v);
                    }
                }
            }
            {
                let rcells: Vec<f32> = self.r.data[idx_min * k..(idx_max + 1) * k].to_vec();
                // scatter_add_O with sign -1 over the block.
                self.scatter_add_o_block(&rcells, k, block_len, &batch_ids_shuf, idx_min, -1.0);
            }

            // E and O stay frozen while every cell in this block is reassigned.
            // logits = assignment_logits(dist_cells, sigma, E, O, theta, block_ids)
            let mut logits = assignment_logits_block(
                &self.dist_mat,
                idx_min,
                idx_max + 1,
                &self.sigma,
                &self.e,
                &self.o,
                &self.theta,
                &batch_ids_shuf,
            );
            Self::normalize_log_assignments(&mut logits).ok_or_else(|| {
                self.numerical_error(
                    "assignment update",
                    "assignment normalizers must be finite and positive",
                )
            })?;
            // Rcells = logits
            let block_start = idx_min * k;
            let block_end = (idx_max + 1) * k;
            self.r.data[block_start..block_end].copy_from_slice(&logits.data);

            // E += sum(Rcells, 1) * Pr_b.t()
            let sums = logits.col_sums(); // K (block length columns)
            for g in 0..k {
                for bidx in 0..self.b {
                    let v = self.e.get(g, bidx) + sums[g] * self.pr_b[bidx];
                    self.e.set(g, bidx, v);
                }
            }
            self.scatter_add_o_block(&logits.data, k, block_len, &batch_ids_shuf, idx_min, 1.0);
        }

        // R = R.cols(reverse_index); dist_mat = dist_mat.cols(reverse_index)
        self.r = self.r.cols(&reverse_index);
        self.dist_mat = self.dist_mat.cols(&reverse_index);
        Ok(())
    }

    /// Scatter-add for a contiguous column block of R (already reordered) into
    /// O, using the shuffled batch ids (port of `scatter_add_O(Rcells,
    /// block_ids, sign)` where Rcells is a submat view).
    fn scatter_add_o_block(
        &mut self,
        rblock: &[f32],
        k: usize,
        block_len: usize,
        ids_shuf: &[u32],
        idx_min: usize,
        sign: f32,
    ) {
        let n = self.n;
        let ncov = self.n_covariates;
        for c in 0..ncov {
            for j in 0..block_len {
                let bidx = ids_shuf[c * n + idx_min + j] as usize;
                let dst = self.o.col_mut(bidx);
                for i in 0..k {
                    let v = rblock[j * k + i];
                    if sign > 0.0 {
                        dst[i] += v;
                    } else {
                        dst[i] -= v;
                    }
                }
            }
        }
    }

    /// Port of `Harmony::check_convergence`.
    fn check_convergence(&self, i_type: u8) -> bool {
        if i_type == 0 {
            if self.objective_kmeans.len() <= self.window_size + 1 {
                return false;
            }
            let mut obj_old = 0.0f32;
            let mut obj_new = 0.0f32;
            let n = self.objective_kmeans.len();
            for i in 0..self.window_size {
                obj_old += self.objective_kmeans[n - 2 - i];
                obj_new += self.objective_kmeans[n - 1 - i];
            }
            return (obj_old - obj_new).abs() / obj_old.abs() < self.epsilon_kmeans;
        }
        if i_type == 1 {
            if self.objective_harmony.len() < 2 {
                return false;
            }
            let obj_old = self.objective_harmony[self.objective_harmony.len() - 2];
            let obj_new = self.objective_harmony[self.objective_harmony.len() - 1];
            return objective_converged(obj_old, obj_new, self.epsilon_harmony);
        }
        true
    }

    /// Port of `Harmony::moe_correct_ridge`.
    fn moe_correct_ridge(&mut self) -> Result<(), HarmonyError> {
        self.z_corr = self.z_orig.clone();
        let multiple_covariates = self.b_vec.len() > 1;
        let k = self.k;
        // Per-batch cell indices, built once (they only depend on the data).
        let batch_cells: Vec<Vec<usize>> = (0..self.b)
            .map(|bidx| self.batch_index[bidx].iter().map(|&v| v as usize).collect())
            .collect();

        for kx in 0..k {
            // avg_R = O.row(k).t() / batch_sizes  (B-length vector)
            let mut avg_r = vec![0.0f32; self.b];
            for bidx in 0..self.b {
                avg_r[bidx] = self.o.get(kx, bidx) / self.batch_sizes[bidx];
            }

            // cov_levels counting.
            let mut cov_levels = vec![0usize; self.b_vec.len()];
            {
                let mut current_cov = 0usize;
                for bidx in 0..self.b {
                    if current_cov < self.covariate_bounds.len()
                        && !(bidx < self.covariate_bounds[current_cov])
                    {
                        current_cov += 1;
                    }
                    if avg_r[bidx] > self.batch_proportion_cutoff {
                        cov_levels[current_cov] += 1;
                    }
                }
            }

            let active_covariates = cov_levels.iter().filter(|&&l| l > 1).count();

            let mut keep: Vec<usize> = Vec::new();
            {
                let mut current_cov = 0usize;
                for bidx in 0..self.b {
                    if current_cov < self.covariate_bounds.len()
                        && !(bidx < self.covariate_bounds[current_cov])
                    {
                        current_cov += 1;
                    }
                    if avg_r[bidx] > self.batch_proportion_cutoff && cov_levels[current_cov] > 1 {
                        keep.push(bidx);
                    }
                }
            }

            if active_covariates == 0 {
                continue;
            }

            let n_keep = keep.len();
            let all_qualify = n_keep == self.b;

            let lamb_vec: Vec<f32> = if all_qualify {
                if self.lambda_estimation {
                    find_lambda(self.alpha, &self.e.row_values(kx))
                } else {
                    self.lambda.clone()
                }
            } else if self.lambda_estimation {
                let esub: Vec<f32> = keep.iter().map(|&bidx| self.e.get(kx, bidx)).collect();
                find_lambda(self.alpha, &esub)
            } else {
                let mut ltmp = vec![0.0f32; n_keep + 1];
                for (i, &bidx) in keep.iter().enumerate() {
                    ltmp[i + 1] = self.lambda[bidx + 1];
                }
                ltmp
            };

            let mat_size = if all_qualify { self.b } else { n_keep } + 1;
            let ok: Vec<f32> = if all_qualify {
                self.o.row_values(kx)
            } else {
                keep.iter().map(|&bidx| self.o.get(kx, bidx)).collect()
            };

            let mut cov_mat = FMat::zeros(mat_size, mat_size);
            {
                let ok_sum: f32 = ok.iter().sum();
                cov_mat.set(0, 0, ok_sum);
                for (i, &v) in ok.iter().enumerate() {
                    cov_mat.set(0, i + 1, v);
                    cov_mat.set(i + 1, 0, v);
                    cov_mat.set(i + 1, i + 1, v);
                }
            }

            // Work on a copy so masking does not change the cluster
            // assignments (Rk = R.row(k) copy).
            let mut rk = vec![0.0f32; self.n];
            for (j, v) in rk.iter_mut().enumerate() {
                *v = self.r.get(kx, j);
            }

            let mut z_sum_all = vec![0.0f32; self.d];
            if multiple_covariates {
                z_sum_all = self.prepare_multi_covariate_ridge(&mut cov_mat, &mut rk, &keep);
            }
            // cov_mat += diagmat(lamb_vec)
            for (i, &l) in lamb_vec.iter().enumerate() {
                let v = cov_mat.get(i, i) + l;
                cov_mat.set(i, i, v);
            }

            let inv_cov: Vec<f32> = if multiple_covariates {
                // Port of arma::inv(cov_mat) — LU with partial pivoting
                // (mat_size is tiny: number of batches + 1).
                inv_lu(&cov_mat)
            } else {
                // Closed-form inverse (matches harmony.cpp's single-covariate
                // Sherman-Morrison-style formula).
                let mut ac = vec![0.0f32; mat_size];
                for i in 0..mat_size {
                    ac[i] = -cov_mat.get(0, i);
                }
                ac[0] = 1.0;
                let b0 = cov_mat.get(0, 0);
                let mut bvec = vec![0.0f32; mat_size];
                for i in 0..mat_size {
                    let d = cov_mat.get(i, i);
                    bvec[i] = if d != 0.0 { 1.0 / d } else { f32::INFINITY };
                }
                bvec[0] = 0.0;
                let mut sq = 0.0f32;
                for i in 0..mat_size {
                    sq += ac[i] * ac[i] * bvec[i];
                }
                let u = b0 - sq;
                let mut ac_b = vec![0.0f32; mat_size];
                for i in 0..mat_size {
                    ac_b[i] = ac[i] * bvec[i];
                }
                ac_b[0] = 1.0;
                let mut inv = vec![0.0f32; mat_size * mat_size];
                for j in 0..mat_size {
                    for i in 0..mat_size {
                        inv[j * mat_size + i] = (1.0 / u) * ac_b[i] * ac_b[j];
                        inv[j * mat_size + i] += if i == j { bvec[i] } else { 0.0 };
                    }
                }
                inv
            };

            let n_batches = if all_qualify { self.b } else { n_keep };

            let mut z_sums: Vec<Vec<f32>> = Vec::with_capacity(n_batches);
            for i in 0..n_batches {
                let bidx = if all_qualify { i } else { keep[i] };
                let idx: Vec<usize> = self.batch_index[bidx].iter().map(|&v| v as usize).collect();
                let rvals: Vec<f32> = idx.iter().map(|&j| rk[j]).collect();
                z_sums.push(self.z_orig.cols_weighted_sum(&idx, &rvals));
                if !multiple_covariates {
                    for dd in 0..self.d {
                        z_sum_all[dd] += z_sums[i][dd];
                    }
                }
            }

            // W = inv_cov.unsafe_col(0) * z_sum_all.t(); then accumulate the
            // per-batch outer products.
            {
                let mut w = FMat::zeros(mat_size, self.d);
                for dd in 0..self.d {
                    let z = z_sum_all[dd];
                    for i in 0..mat_size {
                        w.set(i, dd, inv_cov[i] * z);
                    }
                }
                for i in 0..n_batches {
                    let col = &inv_cov[(i + 1) * mat_size..(i + 2) * mat_size];
                    for dd in 0..self.d {
                        let z = z_sums[i][dd];
                        for row in 0..mat_size {
                            let v = w.get(row, dd) + col[row] * z;
                            w.set(row, dd, v);
                        }
                    }
                }
                self.w = w;
            }

            // Y.col(k) = W.row(0).t(); W.row(0).zeros();
            for dd in 0..self.d {
                self.y.set(dd, kx, self.w.get(0, dd));
                self.w.set(0, dd, 0.0);
            }

            if all_qualify {
                for bidx in 0..self.b {
                    self.correct_block(kx, bidx + 1, &batch_cells[bidx], &rk);
                }
            } else {
                for (i, &bidx) in keep.iter().enumerate() {
                    self.correct_block(kx, i + 1, &batch_cells[bidx], &rk);
                }
            }
        }

        self.y = self.y.normalise_cols2();
        Ok(())
    }

    /// `Z_corr.cols(idx) -= W.row(wrow).t() * Rk.cols(idx)`
    fn correct_block(&mut self, _kx: usize, wrow: usize, idx: &[usize], rk: &[f32]) {
        for &j in idx {
            let r = rk[j];
            for dd in 0..self.d {
                let v = self.z_corr.get(dd, j) - self.w.get(wrow, dd) * r;
                self.z_corr.set(dd, j, v);
            }
        }
    }

    /// Port of `Harmony::prepare_multi_covariate_ridge`.
    fn prepare_multi_covariate_ridge(
        &self,
        cov_mat: &mut FMat,
        weights: &mut [f32],
        keep: &[usize],
    ) -> Vec<f32> {
        // Map retained groups to matrix rows; zero marks an excluded group.
        let mut batch_row = vec![0usize; self.b];
        for (i, &bidx) in keep.iter().enumerate() {
            batch_row[bidx] = i + 1;
        }

        cov_mat.set(0, 0, 0.0);
        for j in 0..self.n {
            let mut selected = false;
            for c in 0..self.n_covariates {
                let row = batch_row[self.batch_ids[c * self.n + j] as usize];
                if row == 0 {
                    continue;
                }
                selected = true;
                for other in c + 1..self.n_covariates {
                    let col = batch_row[self.batch_ids[other * self.n + j] as usize];
                    if col == 0 {
                        continue;
                    }
                    let w = weights[j];
                    let v = cov_mat.get(row, col) + w;
                    cov_mat.set(row, col, v);
                    let v2 = cov_mat.get(col, row) + w;
                    cov_mat.set(col, row, v2);
                }
            }
            // Count each cell once, even if it belongs to several retained
            // groups.
            if selected {
                let v = cov_mat.get(0, 0) + weights[j];
                cov_mat.set(0, 0, v);
            } else {
                weights[j] = 0.0;
            }
        }

        // Zero weights exclude cells without copying their coordinates.
        self.z_orig.mul_vec(weights)
    }
}

/// Port of `find_lambda(alpha, cluster_E)` in harmony.hpp.
pub fn find_lambda(alpha: f32, cluster_e: &[f32]) -> Vec<f32> {
    let mut lambda_vec = vec![0.0f32; cluster_e.len() + 1];
    for (i, &v) in cluster_e.iter().enumerate() {
        lambda_vec[i + 1] = v * alpha;
    }
    lambda_vec
}

/// Port of `objective_converged` in harmony.cpp (exposed as
/// `_objective_converged` in the Python bindings).
pub fn objective_converged(obj_old: f32, obj_new: f32, epsilon: f32) -> bool {
    if !obj_old.is_finite() || !obj_new.is_finite() || !epsilon.is_finite() {
        return false;
    }
    if obj_old == 0.0 {
        return obj_new == 0.0;
    }
    let delta = (obj_old - obj_new) / obj_old.abs();
    delta >= 0.0 && delta < epsilon
}

/// Port of `assignment_logits` in harmony.cpp (full matrix version).
#[allow(clippy::too_many_arguments)]
pub fn assignment_logits(
    distances: &FMat,
    sigma: &[f32],
    e: &FMat,
    o: &FMat,
    theta: &[f32],
    batch_ids: &[u32],
) -> FMat {
    let k = distances.nrows;
    let n = distances.ncols;
    let ncov = batch_ids.len() / n.max(1);

    let mut logits = FMat::zeros(k, n);
    for j in 0..n {
        for i in 0..k {
            logits.set(i, j, -distances.get(i, j) / sigma[i]);
        }
    }

    // log_diversity = log(2E + 1) - log(O + E + 1); each_row *= theta
    // Direct port of the double loop:
    // for c, j: logits.col(j) += log_diversity.col(batch_ids(c, j))
    let mut log_diversity = FMat::zeros(k, e.ncols);
    for bidx in 0..e.ncols {
        for i in 0..k {
            let ld =
                ((2.0 * e.get(i, bidx)) + 1.0).ln() - (o.get(i, bidx) + e.get(i, bidx) + 1.0).ln();
            log_diversity.set(i, bidx, ld * theta[bidx]);
        }
    }
    for c in 0..ncov {
        for j in 0..n {
            let bidx = batch_ids[c * n + j] as usize;
            for i in 0..k {
                let v = logits.get(i, j) + log_diversity.get(i, bidx);
                logits.set(i, j, v);
            }
        }
    }
    logits
}

/// Block variant of `assignment_logits` operating on a contiguous column
/// slice of the reordered distance matrix with shuffled batch ids.
#[allow(clippy::too_many_arguments)]
fn assignment_logits_block(
    dist_reordered: &FMat,
    col_start: usize,
    col_end: usize,
    sigma: &[f32],
    e: &FMat,
    o: &FMat,
    theta: &[f32],
    batch_ids_shuf: &[u32],
) -> FMat {
    let k = dist_reordered.nrows;
    let block_len = col_end - col_start;
    let n = dist_reordered.ncols;
    let ncov = batch_ids_shuf.len() / n.max(1);

    let mut logits = FMat::zeros(k, block_len);
    for j in 0..block_len {
        for i in 0..k {
            logits.set(i, j, -dist_reordered.get(i, col_start + j) / sigma[i]);
        }
    }

    // log_diversity columns for the batch ids appearing in this block.
    let mut log_diversity = FMat::zeros(k, e.ncols);
    for bidx in 0..e.ncols {
        for i in 0..k {
            let ld =
                ((2.0 * e.get(i, bidx)) + 1.0).ln() - (o.get(i, bidx) + e.get(i, bidx) + 1.0).ln();
            log_diversity.set(i, bidx, ld * theta[bidx]);
        }
    }
    for c in 0..ncov {
        for j in 0..block_len {
            let bidx = batch_ids_shuf[c * n + col_start + j] as usize;
            for i in 0..k {
                let v = logits.get(i, j) + log_diversity.get(i, bidx);
                logits.set(i, j, v);
            }
        }
    }
    logits
}

/// Port of `exponentiate_shifted_logits` in harmony.cpp.
/// Mutates `logits` in place: subtract per-column max, exponentiate.
/// Returns the per-column sums (normalizers).
pub fn exponentiate_shifted_logits(logits: &mut FMat) -> Vec<f32> {
    let maxima = logits.col_maxima();
    logits.sub_each_col_from_rows(&maxima);
    logits.exp_in_place();
    logits.row_sums()
}

/// Port of `arma::inv` for small dense f32 matrices — LU with partial
/// pivoting (mirrors Armadillo's default general inverse path closely enough
/// for the differential gates; mat_size is tiny).
fn inv_lu(m: &FMat) -> Vec<f32> {
    let n = m.nrows;
    debug_assert_eq!(m.ncols, n);
    // Build augmented working copy.
    let mut a: Vec<f64> = m.data.iter().map(|&v| v as f64).collect();
    let mut perm: Vec<usize> = (0..n).collect();

    for col in 0..n {
        // Partial pivot.
        let mut pivot = col;
        let mut max_abs = a[col * n + col].abs();
        for row in col + 1..n {
            let v = a[row * n + col].abs();
            if v > max_abs {
                max_abs = v;
                pivot = row;
            }
        }
        if max_abs == 0.0 {
            // Singular matrix: Armadillo would fail; return NaNs to surface
            // downstream as a numerical error.
            return vec![f32::NAN; n * n];
        }
        perm.swap(col, pivot);
        for dd in 0..n {
            a.swap(col * n + dd, pivot * n + dd);
        }
        let d = a[col * n + col];
        for row in col + 1..n {
            let l = a[row * n + col] / d;
            a[row * n + col] = l;
            for dd in col + 1..n {
                let v = a[row * n + dd] - l * a[col * n + dd];
                a[row * n + dd] = v;
            }
        }
    }

    // Solve A * X = I by forward/back substitution per unit vector.
    let mut inv = vec![0.0f64; n * n];
    for jcol in 0..n {
        let mut x = vec![0.0f64; n];
        x[jcol] = 1.0;
        // Apply permutation to b.
        let mut b = vec![0.0f64; n];
        for i in 0..n {
            b[i] = x[perm[i]];
        }
        // Forward substitution (L has unit diagonal).
        for i in 0..n {
            let mut s = b[i];
            for dd in 0..i {
                s -= a[i * n + dd] * b[dd];
            }
            b[i] = s;
        }
        // Back substitution.
        for i in (0..n).rev() {
            let mut s = b[i];
            for dd in i + 1..n {
                s -= a[i * n + dd] * b[dd];
            }
            b[i] = s / a[i * n + i];
        }
        for i in 0..n {
            inv[jcol * n + i] = b[i];
        }
    }
    inv.iter().map(|&v| v as f32).collect()
}
