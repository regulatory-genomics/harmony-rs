//! Port of C++ `std::mt19937` and the libstdc++ distribution algorithms used
//! by the Harmony algorithm.
//!
//! Exact-parity notes (validated against a compiled C++ probe, see
//! `scripts/cpp_probe.cpp` and `tests/rng_reference.rs`):
//!
//! - `std::mt19937`: C++11-specified Mersenne Twister (w=32, n=624, m=397, r=31,
//!   a=0x9908b0df, u=11, d=0xffffffff, s=7, b=0x9d2c5680, t=15, c=0xefc60000,
//!   l=18, f=1812433253), seeded per the C++ standard (`x[0] = seed;
//!   x[i] = f * (x[i-1] ^ (x[i-1] >> 30)) + i`).
//! - `std::uniform_real_distribution<float>(0, 1)` on libstdc++ calls
//!   `generate_canonical<float, 24>`: one u32 draw per call,
//!   `ret = float(u32) / 2^32f` (computed in f32), redrawn while `ret >= 1.0`
//!   (the u32 → f32 conversion rounds up to 2^32 for large u32 values, so the
//!   redraw loop is reachable and must be replicated).
//! - `std::shuffle` on random-access iterators with mt19937 takes the paired
//!   Fisher-Yates branch (`urngrange / urange >= urange`), using
//!   `std::__gen_two_uniform_ints(b0, b1, g)` which draws
//!   `uniform_int_distribution<u64>{0, b0*b1 - 1}(g)` and splits it as
//!   `(x / b1, x % b1)`. The libstdc++ `uniform_int_distribution::operator()`
//!   downscale path for `urngrange == UINT32_MAX` is Lemire's algorithm
//!   (`_S_nd<uint64_t, urbg, uint32_t>`): `product = u64(g()) * u64(range)`,
//!   rejection while `u32(product) < (-range) % range`, result `product >> 32`.

/// Exact port of C++ `std::mt19937` (libstdc++ / C++11 standard).
pub struct Mt19937 {
    mt: [u32; 624],
    idx: usize,
}

impl Mt19937 {
    /// Port of `std::mt19937(seed)`.
    pub fn new(seed: u32) -> Self {
        let mut mt = [0u32; 624];
        mt[0] = seed;
        for i in 1..624 {
            mt[i] = 1812433253u32
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Mt19937 { mt, idx: 624 }
    }

    fn twist(&mut self) {
        for i in 0..624 {
            let y = (self.mt[i] & 0x8000_0000) | (self.mt[(i + 1) % 624] & 0x7fff_ffff);
            self.mt[i] =
                self.mt[(i + 397) % 624] ^ (y >> 1) ^ if y & 1 != 0 { 0x9908_b0df } else { 0 };
        }
        self.idx = 0;
    }

    /// Port of `std::mt19937::operator()` — next raw 32-bit value.
    pub fn next_u32(&mut self) -> u32 {
        if self.idx >= 624 {
            self.twist();
        }
        let mut y = self.mt[self.idx];
        self.idx += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// Port of libstdc++ `generate_canonical<float, 24>` as used by
    /// `std::uniform_real_distribution<float>(0.0f, 1.0f)`.
    ///
    /// One u32 draw per call: `float(u32) / 2^32f` in f32 arithmetic. When the
    /// rounded value reaches 1.0 (raw draw in the top 128 of the u32 range),
    /// libstdc++ clamps to `nextafter(1.0, 0.0)` *without consuming another
    /// draw* — redrawing here would shift the entire downstream stream
    /// (observed at kmeans++ pick 95 on the medium config, raw draw
    /// 4294967281).
    pub fn uniform_real_01(&mut self) -> f32 {
        let u = self.next_u32();
        let ret = (u as f32) / 4294967296.0f32;
        if ret < 1.0 {
            ret
        } else {
            f32::from_bits(0x3f7f_ffff) // nextafter(1.0f, 0.0f)
        }
    }

    /// Port of libstdc++ `uniform_int_distribution<uint32_t>::operator()`
    /// downscale path (`urngrange == UINT32_MAX`): Lemire's algorithm
    /// `_S_nd<uint64_t, Mt19937, uint32_t>` for `uerange` values in [0, range).
    ///
    /// `range` must be in `1..=2^32` (`range == 2^32` is the full-range case,
    /// which the C++ code handles as `uerange` truncation only for ranges that
    /// fit u32; callers never exceed u32 ranges here).
    fn lemire_u32(&mut self, range: u32) -> u32 {
        debug_assert!(range >= 1);
        let mut product = (self.next_u32() as u64) * (range as u64);
        let mut low = product as u32;
        if low < range {
            let threshold = range.wrapping_neg() % range;
            while low < threshold {
                product = (self.next_u32() as u64) * (range as u64);
                low = product as u32;
            }
        }
        (product >> 32) as u32
    }

    /// Port of `std::uniform_int_distribution<uint64_t>{0, b - 1}(g)` as used
    /// by `std::__gen_two_uniform_ints` on libstdc++ with mt19937 (all ranges
    /// here are < 2^32, so the u32-truncated Lemire path applies).
    fn uniform_int_u64(&mut self, uerange: u64) -> u64 {
        debug_assert!(uerange >= 1 && uerange <= u32::MAX as u64);
        self.lemire_u32(uerange as u32) as u64
    }

    /// Port of libstdc++ `std::shuffle` for random-access iterators with
    /// `std::mt19937`. For `urngrange / urange >= urange` (true for all N <
    /// ~2^16 here) it uses the paired Fisher-Yates branch; the generic branch
    /// is replicated for completeness.
    pub fn shuffle(&mut self, slice: &mut [usize]) {
        let n = slice.len();
        if n == 0 {
            return;
        }
        let urngrange: u64 = u32::MAX as u64; // mt19937 max() - min()
        let urange = n as u64;

        if urngrange / urange >= urange {
            // Paired branch: swap positions for two successive elements per
            // distribution invocation.
            let mut i = 1usize;

            // Even number of elements: do the first swap up front with
            // uniform_int_distribution{0, 1} (uerange = 2).
            if n % 2 == 0 {
                let d = self.lemire_u32(2);
                slice.swap(i, d as usize);
                i += 1;
            }

            while i != n {
                // __gen_two_uniform_ints(swap_range, swap_range + 1, g):
                //   swap_range = (i - first) + 1 = i + 1
                //   x = uniform_int_distribution<u64>{0, b0*b1 - 1}(g)
                //   (first, second) = (x / b1, x % b1)
                let b0 = (i + 1) as u64; // swap_range
                let b1 = (i + 2) as u64; // swap_range + 1
                let x = self.uniform_int_u64(b0 * b1);
                let first = x / b1;
                let second = x % b1;
                slice.swap(i, first as usize);
                slice.swap(i + 1, second as usize);
                i += 2;
            }
        } else {
            // Generic branch (unreachable for our N; replicated for parity).
            for i in 1..n {
                let d = self.lemire_u32((i + 1) as u32);
                slice.swap(i, d as usize);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_first_draws_match_cpp_seed_zero() {
        // Reference values from scripts/cpp_probe.cpp (g++, libstdc++),
        // seed 0 — see tests/fixtures/rng_probe_seed0.json.
        let mut rng = Mt19937::new(0);
        let expected: [u32; 8] = [
            2357136044, 2546248239, 3071714933, 3626093760, 2588848963, 3684848379, 2340255427,
            3638918503,
        ];
        for &e in &expected {
            assert_eq!(rng.next_u32(), e);
        }
    }

    #[test]
    fn test_seed_5489_is_cpp_default() {
        // std::mt19937 default-seeded (5489) first draws, from cpp_probe.
        let mut rng = Mt19937::new(5489);
        let expected: [u32; 4] = [3499211612, 581869302, 3890346734, 3586334585];
        for &e in &expected {
            assert_eq!(rng.next_u32(), e);
        }
    }

    #[test]
    fn test_shuffle_large_n_takes_generic_branch() {
        // N = 68785 > 65535, so urngrange < urange^2 and libstdc++
        // std::shuffle takes the generic branch (one draw per swap).
        // Reference values from the shuffle probe (g++ 14, libstdc++):
        // std::shuffle over 0..68785 with std::mt19937(0).
        const N: usize = 68785;
        let mut rng = Mt19937::new(0);
        let mut v: Vec<usize> = (0..N).collect();
        rng.shuffle(&mut v);
        let expected_first8 = [26990usize, 68204, 21904, 3028, 29287, 11021, 25046, 752];
        let expected_last8 = [51379, 39969, 34686, 64699, 45697, 58809, 32163, 2810];
        assert_eq!(&v[..8], &expected_first8[..]);
        assert_eq!(&v[N - 8..], &expected_last8[..]);
        let sum: u64 = v.iter().map(|&x| x as u64).sum();
        assert_eq!(sum, 2365653720);
    }
}
