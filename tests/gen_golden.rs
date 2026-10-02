//! Regenerate the self-golden fixtures from the Rust implementation itself.
//!
//! The algorithm is fully deterministic for a given `random_state`
//! (ChaCha8Rng seeded from it), so these fixtures act as a regression lock-in:
//! `tests/differential.rs` asserts the current run reproduces them exactly.
//!
//! Regenerate (and commit) when an intentional change alters the output:
//!
//!     cargo test --release --test gen_golden -- --ignored --nocapture

mod common;

use common::goldens::{run, write_fixture, Cfg};

#[test]
#[ignore]
fn regenerate_goldens() {
    for cfg in Cfg::all() {
        write_fixture(&run(cfg));
    }
    eprintln!("done");
}
