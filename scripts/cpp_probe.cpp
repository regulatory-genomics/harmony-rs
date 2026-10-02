// C++ probe: generates ground-truth RNG streams from libstdc++ to
// cross-validate src/random.rs (see tests/rng_reference.rs).
//
// Usage: cpp_probe <seed> <n_draws> <shuffle_n> <n_shuffles>
// Dumps JSON to stdout: raw draws, uniform_real draws, and shuffles.
#include <random>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cstdint>

int main(int argc, char** argv) {
    if (argc < 5) { std::cerr << "usage: seed n_draws shuffle_n n_shuffles\n"; return 1; }
    uint32_t seed = std::stoul(argv[1]);
    int n_draws = std::stoi(argv[2]);
    int shuffle_n = std::stoi(argv[3]);
    int n_shuffles = std::stoi(argv[4]);

    std::mt19937 rng(seed);

    std::cout << "{\n  \"seed\": " << seed << ",\n";

    std::cout << "  \"raw\": [";
    for (int i = 0; i < n_draws; ++i) {
        if (i) std::cout << ", ";
        std::cout << rng();
    }
    std::cout << "],\n";

    std::cout << "  \"uniform\": [";
    {
        std::mt19937 rng2(seed);
        std::uniform_real_distribution<float> uniform01(0.0f, 1.0f);
        for (int i = 0; i < n_draws; ++i) {
            if (i) std::cout << ", ";
            std::cout << "\"" << std::hexfloat << uniform01(rng2) << std::defaultfloat << "\"";
        }
    }
    std::cout << "],\n";

    std::cout << "  \"shuffles\": [";
    for (int s = 0; s < n_shuffles; ++s) {
        std::mt19937 rng3(seed + static_cast<uint32_t>(s));
        std::vector<unsigned> v(shuffle_n);
        for (int i = 0; i < shuffle_n; ++i) v[i] = i;
        std::shuffle(v.begin(), v.end(), rng3);
        if (s) std::cout << ", ";
        std::cout << "\n    [";
        for (int i = 0; i < shuffle_n; ++i) {
            if (i) std::cout << ", ";
            std::cout << v[i];
        }
        std::cout << "]";
    }
    std::cout << "\n  ]\n}\n";
    return 0;
}
