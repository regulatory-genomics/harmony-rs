//! Shared test helpers — ports of `read_tsv`, `cols_to_matrix`, and
//! `pearsonr` from `../harmonypy/tests/test_harmony.py`.
//!
//! Not every helper is used by every test binary; unused ones are allowed.

#![allow(dead_code)]

use std::collections::HashMap;
use std::io::Read;

/// Port of `read_tsv(path)` — returns (header, columns_dict). Numeric columns
/// are f64 arrays; non-numeric columns are strings.
pub fn read_tsv(path: &str) -> (Vec<String>, HashMap<String, Column>) {
    let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("failed to open {path}: {e}"));
    let mut reader = if path.ends_with(".gz") {
        Box::new(flate2::read::GzDecoder::new(file)) as Box<dyn Read>
    } else {
        Box::new(file) as Box<dyn Read>
    };
    let mut text = String::new();
    reader.read_to_string(&mut text).unwrap();

    let mut lines = text.lines();
    let header: Vec<String> = lines
        .next()
        .expect("empty tsv")
        .split('\t')
        .map(|s| s.to_string())
        .collect();
    let n_cols = header.len();
    let rows: Vec<Vec<&str>> = lines
        .filter(|l| !l.is_empty())
        .map(|l| {
            let mut parts: Vec<&str> = l.split('\t').collect();
            parts.resize(n_cols, "");
            parts
        })
        .collect();

    let mut cols: HashMap<String, Column> = HashMap::new();
    for (i, name) in header.iter().enumerate() {
        let values: Vec<&str> = rows.iter().map(|r| r[i]).collect();
        // Try float64 first (mirrors np.array(values, dtype=np.float64)).
        let mut all_numeric = true;
        let mut floats = Vec::with_capacity(values.len());
        for v in &values {
            match parse_float(v) {
                Some(f) => floats.push(f),
                None => {
                    all_numeric = false;
                    break;
                }
            }
        }
        if all_numeric {
            cols.insert(name.clone(), Column::Float(floats));
        } else {
            cols.insert(
                name.clone(),
                Column::String(values.iter().map(|s| s.to_string()).collect()),
            );
        }
    }
    (header, cols)
}

/// Parse a float the way Python's float() does (empty/NaN strings handled).
fn parse_float(s: &str) -> Option<f64> {
    let t = s.trim();
    if t.is_empty() {
        return None;
    }
    t.parse::<f64>().ok()
}

#[derive(Clone, Debug)]
pub enum Column {
    Float(Vec<f64>),
    String(Vec<String>),
}

impl Column {
    pub fn floats(&self) -> &[f64] {
        match self {
            Column::Float(v) => v,
            Column::String(_) => panic!("expected float column"),
        }
    }

    pub fn strings(&self) -> &[String] {
        match self {
            Column::String(v) => v,
            Column::Float(_) => panic!("expected string column"),
        }
    }
}

/// Port of `cols_to_matrix(cols, header)` — extract numeric columns into an
/// N x d row-major matrix.
pub fn cols_to_matrix(
    cols: &HashMap<String, Column>,
    header: &[String],
) -> (Vec<f64>, usize, usize) {
    let mut numeric: Vec<&Column> = Vec::new();
    for name in header {
        if let Some(col) = cols.get(name) {
            if matches!(col, Column::Float(_)) {
                numeric.push(col);
            }
        }
    }
    let n = numeric.first().map(|c| c.floats().len()).unwrap_or(0);
    let d = numeric.len();
    let mut out = vec![0.0; n * d];
    for (j, col) in numeric.iter().enumerate() {
        for (i, &v) in col.floats().iter().enumerate() {
            out[i * d + j] = v;
        }
    }
    (out, n, d)
}

/// Port of `pearsonr(x, y)` in test_harmony.py.
pub fn pearsonr(x: &[f64], y: &[f64]) -> f64 {
    let xm = mean(x);
    let ym = mean(y);
    let mut num = 0.0;
    let mut dx = 0.0;
    let mut dy = 0.0;
    for i in 0..x.len() {
        let a = x[i] - xm;
        let b = y[i] - ym;
        num += a * b;
        dx += a * a;
        dy += b * b;
    }
    num / (dx.sqrt() * dy.sqrt())
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}
