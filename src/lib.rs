//! Rust implementation of the Harmony batch-effect correction algorithm for
//! single-cell data, following the C++/Armadillo backend step-by-step (which
//! itself follows the R harmony2 package).
//!
//! Module mapping:
//! - `run_harmony` wrapper → [`api`]
//! - Harmony core → [`harmony::core`]
//! - k-means initialization → [`harmony::kmeans`]
//! - LISI → [`lisi`]
//! - MT19937 + libstdc++ distributions → [`random`]
//! - f32 matrix operations → [`fmat`]
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
