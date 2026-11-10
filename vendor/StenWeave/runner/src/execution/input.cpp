#include "execution/input.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

namespace stencil {
auto input_value(std::uint64_t seed, std::uint64_t index) -> double {
    std::uint64_t z = seed + 0x9e3779b97f4a7c15ULL * (index + 1);
    z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
    z ^= z >> 31U;

    double const unit = static_cast<double>(z >> 11U) * 0x1.0p-53;
    return unit * 200.0 - 100.0;
}

auto generate_input(std::uint64_t seed, std::size_t size, unsigned threads) -> std::vector<double> {
    std::vector<double> result(size);

    if (!threads) {
        threads = std::max(1U, std::thread::hardware_concurrency());
    }
    threads = static_cast<unsigned>(std::min<std::size_t>(threads, std::max<std::size_t>(1, size)));

    std::vector<std::thread> workers;
    // Join already-created workers if thread creation fails.
    try {
        for (unsigned t = 0; t < threads; ++t) {
            std::size_t const begin = size / threads * t + std::min<std::size_t>(t, size % threads);
            std::size_t const end = begin + size / threads + (t < size % threads ? 1 : 0);

            workers.emplace_back([&, begin, end] {
                for (std::size_t i = begin; i < end; ++i) {
                    result[i] = input_value(seed, i);
                }
            });
        }
    } catch (...) {
        for (std::thread &worker : workers) {
            worker.join();
        }

        throw;
    }

    for (std::thread &worker : workers) {
        worker.join();
    }

    return result;
}
}  // namespace stencil
