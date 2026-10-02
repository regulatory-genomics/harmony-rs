//! Self-golden regression tests: the committed fixtures under
//! `tests/fixtures/differential_*.json` are generated from the Rust
//! implementation itself (see `tests/gen_golden.rs`), because the algorithm is
//! deterministic for a given `random_state` (ChaCha8Rng).
//!
//! Each test:
//! - [`check_exact`] — reproduces the committed fixture bit-for-bit (lock-in
//!   against algorithmic regressions).
//! - [`check_quality`] — statistical sanity (finite outputs, variance
//!   preservation, k-means progress, LISI batch mixing not degrading).
//!
//! These replace the former differential-vs-Python gates; they validate that
//! Harmony converges to a high-quality, reproducible correction rather than
//! matching a specific (RNG-specific) oracle output bit-for-bit.

#![allow(clippy::needless_range_loop)]

mod common;

use common::goldens::{check_exact, check_quality, fixture_path, run, Cfg};
use serde_json::Value;

fn load(cfg: Cfg) -> Value {
    let text = std::fs::read_to_string(fixture_path(cfg.name())).unwrap();
    serde_json::from_str(&text).unwrap()
}

#[test]
fn differential_small_defaults() {
    let r = run(Cfg::SmallDefaults);
    check_exact(&r, &load(Cfg::SmallDefaults));
    check_quality(&r);
}

#[test]
fn differential_iter2() {
    let r = run(Cfg::Iter2);
    check_exact(&r, &load(Cfg::Iter2));
    check_quality(&r);
}

#[test]
fn differential_fixed_lamb() {
    let r = run(Cfg::FixedLamb);
    check_exact(&r, &load(Cfg::FixedLamb));
    check_quality(&r);
}

#[test]
fn differential_multi_cov() {
    let r = run(Cfg::MultiCov);
    check_exact(&r, &load(Cfg::MultiCov));
    check_quality(&r);
}

#[test]
fn differential_medium() {
    let r = run(Cfg::Medium);
    check_exact(&r, &load(Cfg::Medium));
    check_quality(&r);
}
