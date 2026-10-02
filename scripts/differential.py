"""Generate Python-oracle golden fixtures for the differential parity tests.

Run from /tmp/opencode (the repo at /home/kzhang/xxx/harmonypy shadows the
installed harmonypy wheel):

    python3 /home/kzhang/xxx/harmonypy-rs/scripts/differential.py

Writes tests/fixtures/differential_<name>.json with the Python reference
outputs (Z_corr, Y, R, objectives, kmeans_rounds). Configs:

  small_defaults  pbmc_3500 + donor, Python defaults (10 iters)
  iter2           pbmc_3500 + donor, max_iter_harmony=2, max_iter_kmeans=2
                  (exercises the iteration-2+ re-init path)
  fixed_lamb      pbmc_3500 + donor, lamb=1.0 (fixed-lambda path)
  multi_cov       synthetic 400 cells x 5 PCs, lab (2) + day (3) covariates
                  (inputs dumped alongside the outputs)
  medium          ircolitis + batch (69k cells), max_iter_harmony=1
                  (r skipped: N x K would be ~700 MB of JSON)

The TSVs are parsed manually (float() per field) because pandas is not
installed in this environment; float() and Rust's f64 parse are both
correctly rounded, so the parsed matrices are bit-identical.
"""
import gzip
import json
import sys

import numpy as np

import harmonypy

DATA = "/home/kzhang/xxx/harmonypy/data"
OUT = "/home/kzhang/xxx/harmonypy-rs/tests/fixtures"


def read_tsv(path):
    with gzip.open(path, "rt") as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]
    header = lines[0].split("\t")
    rows = [l.split("\t") for l in lines[1:]]
    cols = {}
    for i, name in enumerate(header):
        values = [r[i] for r in rows]
        try:
            cols[name] = np.array([float(v) for v in values], dtype=np.float64)
        except ValueError:
            cols[name] = np.array(values, dtype=object)
    return cols


def pcs_matrix(pcs):
    pc_names = [k for k, v in pcs.items() if v.dtype == np.float64]
    X = np.column_stack([pcs[k] for k in pc_names])
    return X, pc_names


def dump(name, ho, include_r=True, inputs=None):
    out = {
        "z_corr": ho.Z_corr.tolist(),          # N x d
        "y": ho.Y.tolist(),                    # d x K
        "objective_kmeans": list(ho.objective_kmeans),
        "objective_harmony": list(ho.objective_harmony),
        "kmeans_rounds": list(ho.kmeans_rounds),
        "K": int(ho.K),
    }
    if include_r:
        out["r"] = ho.R.tolist()               # N x K
    if inputs is not None:
        out["inputs"] = inputs
    with open(f"{OUT}/differential_{name}.json", "w") as f:
        json.dump(out, f)
    size = len(out["z_corr"])
    print(f"{name}: N={size} K={out['K']} rounds={out['kmeans_rounds']} "
          f"obj_k={[round(v, 4) for v in out['objective_kmeans']]}")


# --- small_defaults: pbmc_3500 + donor, Python defaults ---------------------
meta = read_tsv(f"{DATA}/pbmc_3500_meta.tsv.gz")
pcs = read_tsv(f"{DATA}/pbmc_3500_pcs.tsv.gz")
X, _ = pcs_matrix(pcs)
ho = harmonypy.run_harmony(X, meta, ["donor"], verbose=False)
dump("small_defaults", ho)

# --- iter2: exercises the iteration-2+ re-init path --------------------------
ho = harmonypy.run_harmony(X, meta, ["donor"], max_iter_harmony=2,
                           max_iter_kmeans=2, verbose=False)
dump("iter2", ho)

# --- fixed_lamb: fixed ridge penalty -----------------------------------------
ho = harmonypy.run_harmony(X, meta, ["donor"], lamb=1.0, verbose=False)
dump("fixed_lamb", ho)

# --- multi_cov: synthetic 400 cells, lab + day covariates --------------------
rng = np.random.RandomState(42)
n_cells, n_pcs = 400, 5
Xm = rng.randn(n_cells, n_pcs)
lab_labels = np.array(["lab_a", "lab_b"])
day_labels = np.array(["Monday", "Tuesday", "Wednesday"])
lab = lab_labels[rng.randint(0, 2, n_cells)]
day = day_labels[rng.randint(0, 3, n_cells)]
ho = harmonypy.run_harmony(
    Xm, {"lab": lab, "day": day}, ["lab", "day"],
    max_iter_harmony=2, max_iter_kmeans=2, verbose=False,
)
dump("multi_cov", ho, inputs={
    "x": Xm.tolist(),           # N x d, row-major
    "lab": lab.tolist(),
    "day": day.tolist(),
})

# --- medium: ircolitis + batch, 69k cells -------------------------------------
meta_m = read_tsv(f"{DATA}/ircolitis_blood_cd8_obs.tsv.gz")
pcs_m = read_tsv(f"{DATA}/ircolitis_blood_cd8_pcs.tsv.gz")
Xm2, _ = pcs_matrix(pcs_m)
ho = harmonypy.run_harmony(Xm2, meta_m, ["batch"], max_iter_harmony=1,
                           verbose=False)
dump("medium", ho, include_r=False)

print("done")
