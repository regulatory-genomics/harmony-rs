//! Rust implementation of the Harmony batch-effect correction algorithm for
//! single-cell data, following the C++/Armadillo backend step-by-step (which
//! itself follows the R harmony2 package).
//!
//! Module mapping:
//! - `run_harmony` wrapper → [`api`]
//! - Harmony core → [`harmony::core`]
//! - k-means initialization → [`harmony::kmeans`]
//! - LISI → [`lisi`]
//! - f32 matrix operations → [`fmat`]
//!
//! Determinism: randomness uses `rand_chacha::ChaCha8Rng` seeded from the
//! user-supplied `random_state` (`0` by default), so results are fully
//! reproducible for a given seed.
//!
//! See `docs/parity.md` for the parity contract.

pub mod api;
pub mod fmat;
pub mod harmony;
pub mod lisi;

pub use api::{compute_lisi, run_harmony, HarmonyResult, RunHarmonyArgs, ScalarOrVec};
pub use harmony::core::{Harmony, HarmonyError};
pub use lisi::{compute_lisi_impl, LisiError};
