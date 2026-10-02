//! Port of the Python `harmonypy` package v2.0.2 (../harmonypy).
//!
//! harmonypy is a Rust reimplementation of the Harmony batch-effect correction
//! algorithm for single-cell data, following the C++ backend in
//! `../harmonypy/src/harmony.cpp` step-by-step (which itself follows the R
//! harmony2 package).
//!
//! Module mapping (Python/C++ → Rust):
//! - `harmonypy/harmony.py` (`run_harmony` wrapper) → [`api`]
//! - `src/harmony.cpp` (Harmony core) → [`harmony::core`]
//! - `src/harmony.cpp` (`kmeans_init`) → [`harmony::kmeans`]
//! - `src/lisi.hpp` (LISI) → [`lisi`]
//! - `std::mt19937` + libstdc++ distributions → [`random`]
//! - `arma::fmat` operations → [`fmat`]
//!
//! See `docs/parity.md` for the parity contract.

pub mod api;
pub mod fmat;
pub mod harmony;
pub mod lisi;
pub mod random;

pub use api::{compute_lisi, run_harmony, HarmonyResult, RunHarmonyArgs, ScalarOrVec};
pub use harmony::core::{Harmony, HarmonyError};
pub use lisi::{compute_lisi_impl, LisiError};
pub use random::Mt19937;
