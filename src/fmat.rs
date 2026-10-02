//! Port of the Armadillo `arma::fmat` (MATTYPE) operations used by
//! `../harmonypy/src/harmony.cpp`.
//!
//! Column-major f32 matrices, mirroring `typedef arma::Mat<float> MATTYPE`.
//! Only the operations the Harmony algorithm uses are implemented; products
//! use plain f32 accumulation loops (the reference delegates to OpenBLAS, so
//! results agree within the differential gates — see docs/parity.md).
//!
//! Index-based loops are intentional throughout: they mirror the C++
//! reference's flat-buffer (column-major) indexing directly.

#![allow(clippy::needless_range_loop)]

/// Threshold (in FLOPs) below which matrix products use plain f32 loops
/// instead of `faer` — faer's parallel dispatch (rayon thread-pool startup
/// and sync) costs more than the arithmetic itself for small problems.
pub const FAER_MIN_FLOPS: usize = 10_000_000;

/// Threshold (in elements scanned) below which streaming ops (gemv-style
/// products, distance loops) use plain f32 loops. These are memory-bound,
/// so `faer`'s threading already pays off at much smaller sizes than for
/// gemm.
pub const FAER_MIN_ELEMS: usize = 2_000_000;

/// Column-major f32 matrix (port of `arma::fmat` usage in harmony.cpp).
#[derive(Clone, Debug)]
pub struct FMat {
    pub nrows: usize,
    pub ncols: usize,
    pub data: Vec<f32>,
}

impl FMat {
    /// Port of `arma::fmat(r, c, arma::fill::zeros)`.
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        FMat {
            nrows,
            ncols,
            data: vec![0.0; nrows * ncols],
        }
    }

    /// Port of `arma::fmat(r, c)` from a column-major slice.
    pub fn from_col_major(nrows: usize, ncols: usize, data: Vec<f32>) -> Self {
        debug_assert_eq!(data.len(), nrows * ncols);
        FMat { nrows, ncols, data }
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f32 {
        self.data[j * self.nrows + i]
    }

    #[inline]
    pub fn set(&mut self, i: usize, j: usize, v: f32) {
        self.data[j * self.nrows + i] = v;
    }

    #[inline]
    pub fn col(&self, j: usize) -> &[f32] {
        &self.data[j * self.nrows..(j + 1) * self.nrows]
    }

    #[inline]
    pub fn col_mut(&mut self, j: usize) -> &mut [f32] {
        let n = self.nrows;
        &mut self.data[j * n..(j + 1) * n]
    }

    /// Set column j from a slice (port of `Y.col(i) = X.col(idx)`).
    pub fn set_col(&mut self, j: usize, src: &[f32]) {
        self.col_mut(j).copy_from_slice(src);
    }

    /// Port of `X.row(i)` — copy of row i (length ncols).
    pub fn row_values(&self, i: usize) -> Vec<f32> {
        (0..self.ncols).map(|j| self.get(i, j)).collect()
    }

    /// Port of `arma::conv_to<MATTYPE>::from(Z)` — f64 → f32 per element,
    /// from a row-major f64 input of shape (nrows, ncols).
    pub fn from_row_major_f64(nrows: usize, ncols: usize, data: &[f64]) -> Self {
        let mut out = vec![0.0f32; nrows * ncols];
        for j in 0..ncols {
            for i in 0..nrows {
                out[j * nrows + i] = data[i * ncols + j] as f32;
            }
        }
        FMat {
            nrows,
            ncols,
            data: out,
        }
    }

    /// Extract a set of columns into a new matrix (port of `X.cols(idx)`).
    pub fn cols(&self, idx: &[usize]) -> FMat {
        let mut out = FMat::zeros(self.nrows, idx.len());
        for (j, &i) in idx.iter().enumerate() {
            out.set_col(j, self.col(i));
        }
        out
    }

    /// Port of `arma::normalise(X, 2, 0)` — divide each column by its 2-norm.
    /// Zero-norm columns are left unchanged (Armadillo skips them).
    pub fn normalise_cols2(&self) -> FMat {
        let mut out = self.clone();
        for j in 0..self.ncols {
            let col = self.col(j);
            // f64 accumulation for the norm (Armadillo uses BLAS snrm2 in the
            // reference; f64 accumulation stays within the differential gates).
            let norm: f32 =
                (col.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>()).sqrt() as f32;
            if norm > 0.0 && norm.is_finite() {
                for v in out.col_mut(j) {
                    *v /= norm;
                }
            }
        }
        out
    }

    /// Port of `arma::accu(X)` — sum of all elements (column-major order),
    /// using Armadillo's `arrayops::accumulate` accumulation order: two
    /// interleaved partial sums (even-indexed elements in acc1, odd in acc2),
    /// combined as acc1 + acc2.
    pub fn accu(&self) -> f32 {
        let n = self.data.len();
        let mut acc1 = 0.0f32;
        let mut acc2 = 0.0f32;
        let mut i = 0usize;
        while i + 1 < n {
            acc1 += self.data[i];
            acc2 += self.data[i + 1];
            i += 2;
        }
        if i < n {
            acc1 += self.data[i];
        }
        acc1 + acc2
    }

    /// Port of `arma::accu(R % dist_mat)` — elementwise product sum. Both
    /// operands are plain matrices, so the C++ takes the `op_dot::direct_dot`
    /// path (OpenBLAS sdot); we compute products in column-major order and
    /// apply Armadillo's dual-accumulator accumulation order (bitwise match
    /// is impossible against a threaded BLAS; see docs/parity.md).
    pub fn accu_product(&self, other: &FMat) -> f32 {
        debug_assert_eq!(self.data.len(), other.data.len());
        let n = self.data.len();
        let mut acc1 = 0.0f32;
        let mut acc2 = 0.0f32;
        let mut i = 0usize;
        while i + 1 < n {
            acc1 += self.data[i] * other.data[i];
            acc2 += self.data[i + 1] * other.data[i + 1];
            i += 2;
        }
        if i < n {
            acc1 += self.data[i] * other.data[i];
        }
        acc1 + acc2
    }

    /// Port of `arma::sum(X, 0)` — per-column sums (length ncols), each
    /// column accumulated with Armadillo's dual-accumulator
    /// `arrayops::accumulate` order.
    pub fn row_sums(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; self.ncols];
        for j in 0..self.ncols {
            out[j] = Self::accumulate(&self.data[j * self.nrows..(j + 1) * self.nrows]);
        }
        out
    }

    /// Port of `arrayops::accumulate` — two interleaved partial sums.
    fn accumulate(src: &[f32]) -> f32 {
        let n = src.len();
        let mut acc1 = 0.0f32;
        let mut acc2 = 0.0f32;
        let mut i = 0usize;
        while i + 1 < n {
            acc1 += src[i];
            acc2 += src[i + 1];
            i += 2;
        }
        if i < n {
            acc1 += src[i];
        }
        acc1 + acc2
    }

    /// Port of `arma::sum(X, 1)` — per-row sums (length nrows).
    pub fn col_sums(&self) -> Vec<f32> {
        self.cols_range_sums(0, self.ncols)
    }

    /// Per-row sums of columns [j0, j1) (port of `arma::sum(Rcells, 1)` —
    /// length nrows).
    pub fn cols_range_sums(&self, j0: usize, j1: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; self.nrows];
        for j in j0..j1 {
            for (i, v) in self.col(j).iter().enumerate() {
                out[i] += v;
            }
        }
        out
    }

    /// Port of `arma::max(X, 0)` — per-column maxima.
    pub fn col_maxima(&self) -> Vec<f32> {
        let mut out = vec![f32::NEG_INFINITY; self.ncols];
        for j in 0..self.ncols {
            for &v in self.col(j) {
                if v > out[j] {
                    out[j] = v;
                }
            }
        }
        out
    }

    /// Port of `X.min()` — global minimum (NaN handling: Armadillo's min
    /// propagates NaN per IEEE comparisons; we mirror arma's is_finite check
    /// usage by returning f32::MIN on all-NaN only if that matters — here
    /// callers check is_finite first).
    pub fn min(&self) -> f32 {
        let mut m = f32::INFINITY;
        for &v in &self.data {
            if v < m {
                m = v;
            }
        }
        m
    }

    pub fn is_finite(&self) -> bool {
        self.data.iter().all(|v| v.is_finite())
    }

    /// Port of `Y.t() * Z` for d×K times d×N → K×N.
    /// Port of `Y.t() * Z_corr` — d×K transposed times d×N → K×N matrix.
    /// Large products run through `faer` (SIMD + rayon-parallel), matching
    /// the reference's OpenBLAS-backed sgemm in role; small ones use plain
    /// f32 loops. Results agree within the differential gates (see
    /// docs/parity.md, tolerated variation #1).
    pub fn mul_trans_a(&self, z: &FMat) -> FMat {
        debug_assert_eq!(self.nrows, z.nrows);
        if self.ncols * z.ncols * self.nrows >= FAER_MIN_FLOPS {
            let a = faer::MatRef::from_column_major_slice(&self.data, self.nrows, self.ncols);
            let b = faer::MatRef::from_column_major_slice(&z.data, z.nrows, z.ncols);
            let c: faer::Mat<f32> = a.transpose() * b;
            let mut out = FMat::zeros(self.ncols, z.ncols);
            for j in 0..z.ncols {
                out.col_mut(j).copy_from_slice(c.col_as_slice(j));
            }
            out
        } else {
            let mut out = FMat::zeros(self.ncols, z.ncols);
            for j in 0..z.ncols {
                let zcol = z.col(j);
                for g in 0..self.ncols {
                    let ycol = self.col(g);
                    let mut acc = 0.0f32;
                    for i in 0..self.nrows {
                        acc += ycol[i] * zcol[i];
                    }
                    out.set(g, j, acc);
                }
            }
            out
        }
    }

    /// Port of `Z_orig * weights.t()` — d×N times N×1 → d-length vector.
    pub fn mul_vec(&self, w: &[f32]) -> Vec<f32> {
        debug_assert_eq!(self.ncols, w.len());
        if self.ncols * self.nrows >= FAER_MIN_ELEMS {
            let a = faer::MatRef::from_column_major_slice(&self.data, self.nrows, self.ncols);
            let v = faer::MatRef::from_column_major_slice(w, w.len(), 1);
            let c: faer::Mat<f32> = a * v;
            let mut out = vec![0.0f32; self.nrows];
            out.copy_from_slice(c.col_as_slice(0));
            out
        } else {
            let mut out = vec![0.0f32; self.nrows];
            for j in 0..self.ncols {
                let wj = w[j];
                for (i, &v) in self.col(j).iter().enumerate() {
                    out[i] += v * wj;
                }
            }
            out
        }
    }

    /// Port of `Z_orig.cols(idx) * Rk.cols(idx).t()` — d×m times m×1
    /// → d-length vector (weighted column sums over selected columns).
    pub fn cols_weighted_sum(&self, idx: &[usize], w: &[f32]) -> Vec<f32> {
        debug_assert_eq!(idx.len(), w.len());
        let mut out = vec![0.0f32; self.nrows];
        for (&j, &wj) in idx.iter().zip(w.iter()) {
            if wj == 0.0 {
                continue;
            }
            for (i, &v) in self.col(j).iter().enumerate() {
                out[i] += v * wj;
            }
        }
        out
    }

    /// Port of `X.each_col() /= v` — divide each column elementwise by the
    /// column-indexed vector (here: each row i scaled by v[i], matching
    /// `logits.each_col() /= sigma` where sigma indexes rows).
    pub fn div_each_col_by_row_vec(&mut self, v: &[f32]) {
        debug_assert_eq!(self.nrows, v.len());
        for j in 0..self.ncols {
            for (i, x) in self.col_mut(j).iter_mut().enumerate() {
                *x /= v[i];
            }
        }
    }

    /// Port of `X.each_col() += col` — add a column vector to every column.
    pub fn add_each_col(&mut self, col: &[f32]) {
        debug_assert_eq!(self.nrows, col.len());
        for j in 0..self.ncols {
            for (i, x) in self.col_mut(j).iter_mut().enumerate() {
                *x += col[i];
            }
        }
    }

    /// Port of `X.each_col() %= v` — scale row i by v[i] (v indexes rows).
    pub fn mul_each_col_by_row_vec(&mut self, v: &[f32]) {
        debug_assert_eq!(self.nrows, v.len());
        for j in 0..self.ncols {
            for (i, x) in self.col_mut(j).iter_mut().enumerate() {
                *x *= v[i];
            }
        }
    }

    /// Port of `X.each_row() %= v` — scale column j by v[j] (v indexes
    /// columns).
    pub fn mul_each_row_by_col_vec(&mut self, v: &[f32]) {
        debug_assert_eq!(self.ncols, v.len());
        for j in 0..self.ncols {
            for x in self.col_mut(j) {
                *x *= v[j];
            }
        }
    }

    /// Port of `X.each_row() -= v` — subtract per-column vector from each row
    /// (`logits.each_row() -= max(logits, 0)` subtracts col_max[j] from every
    /// row of column j).
    pub fn sub_each_col_from_rows(&mut self, v: &[f32]) {
        debug_assert_eq!(self.ncols, v.len());
        for j in 0..self.ncols {
            for x in self.col_mut(j) {
                *x -= v[j];
            }
        }
    }

    /// Port of `X.each_row() /= v` — divide each row's column j by v[j].
    pub fn div_rows_by_col_vec(&mut self, v: &[f32]) {
        debug_assert_eq!(self.ncols, v.len());
        for j in 0..self.ncols {
            for x in self.col_mut(j) {
                *x /= v[j];
            }
        }
    }

    /// Port of `X = arma::exp(X)` elementwise.
    pub fn exp_in_place(&mut self) {
        for v in &mut self.data {
            *v = v.exp();
        }
    }
}
