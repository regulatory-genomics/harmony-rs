//! Cross-validation of `random.rs` against compiled C++ ground truth
//! (`scripts/cpp_probe.cpp`, g++/libstdc++). The probe output JSON fixtures in
//! `tests/fixtures/rng_probe_*.json` were generated with:
//!
//!   g++ -O2 -std=c++17 scripts/cpp_probe.cpp -o /tmp/cpp_probe
//!   /tmp/cpp_probe <seed> <n_draws> <shuffle_n> <n_shuffles> > fixture.json
//!
//! These tests require an exact match: any divergence in the mt19937 stream,
//! generate_canonical, or std::shuffle semantics is a parity failure.

use harmonypy::Mt19937;
use serde_json::Value;

fn load_probe(name: &str) -> Value {
    let path = format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), name);
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("failed to read {path}: {e}"));
    serde_json::from_str(&text).unwrap()
}

#[test]
fn test_raw_stream_matches_cpp() {
    for name in [
        "rng_probe_seed0.json",
        "rng_probe_seed5489.json",
        "rng_probe_seed42.json",
    ] {
        let probe = load_probe(name);
        let seed = probe["seed"].as_u64().unwrap() as u32;
        let mut rng = Mt19937::new(seed);
        for (i, expected) in probe["raw"].as_array().unwrap().iter().enumerate() {
            assert_eq!(
                rng.next_u32(),
                expected.as_u64().unwrap() as u32,
                "{name}: raw draw {i} mismatch"
            );
        }
    }
}

#[test]
fn test_uniform_real_matches_cpp() {
    for name in [
        "rng_probe_seed0.json",
        "rng_probe_seed5489.json",
        "rng_probe_seed42.json",
    ] {
        let probe = load_probe(name);
        let seed = probe["seed"].as_u64().unwrap() as u32;
        let mut rng = Mt19937::new(seed);
        for (i, expected) in probe["uniform"].as_array().unwrap().iter().enumerate() {
            // Probe values are printed with hexfloat — exact f32 bit patterns.
            let hex = expected.as_str().unwrap();
            let expected_bits = parse_hexfloat_f32_bits(hex);
            let actual = rng.uniform_real_01();
            assert_eq!(
                actual.to_bits(),
                expected_bits,
                "{name}: uniform draw {i} mismatch: {actual} vs {hex}"
            );
        }
    }
}

#[test]
fn test_shuffle_matches_cpp() {
    for name in [
        "rng_probe_seed0.json",
        "rng_probe_seed5489.json",
        "rng_probe_seed42.json",
    ] {
        let probe = load_probe(name);
        let seed = probe["seed"].as_u64().unwrap() as u32;
        let expected_shuffles = probe["shuffles"].as_array().unwrap();
        assert!(!expected_shuffles.is_empty());
        for (s, expected) in expected_shuffles.iter().enumerate() {
            let n = expected.as_array().unwrap().len();
            let mut v: Vec<usize> = (0..n).collect();
            let mut rng = Mt19937::new(seed + s as u32);
            rng.shuffle(&mut v);
            let expected: Vec<usize> = expected
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect();
            assert_eq!(v, expected, "{name}: shuffle {s} mismatch");
        }
    }
}

/// Parse a C++ hexfloat literal like `0x1.18fe16p-1` into exact f32 bits.
/// The probe prints float values; the hexfloat has enough digits to
/// round-trip f32 exactly.
fn parse_hexfloat_f32_bits(hex: &str) -> u32 {
    let s = hex.trim_start_matches("0x");
    let (mantissa_str, exp_str) = s.split_once('p').unwrap();
    let exp: i32 = exp_str.parse().unwrap();
    let (int_part, frac_part) = mantissa_str.split_once('.').unwrap_or((mantissa_str, ""));
    let int_val: f64 = u64::from_str_radix(int_part, 16).unwrap() as f64;
    let mut frac_val = 0.0f64;
    let mut scale = 1.0f64 / 16.0;
    for c in frac_part.chars() {
        frac_val += c.to_digit(16).unwrap() as f64 * scale;
        scale /= 16.0;
    }
    let value = (int_val + frac_val) * (exp as f64).exp2();
    // f32 round-trip: the printed hexfloat for a float has 13 hex digits
    // (enough for double precision); rounding to f32 recovers the exact bits.
    let f = value as f32;
    assert_eq!(f as f64 as f32, f);
    f.to_bits()
}
