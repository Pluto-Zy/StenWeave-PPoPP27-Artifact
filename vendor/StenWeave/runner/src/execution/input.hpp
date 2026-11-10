#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace stencil {
auto input_value(std::uint64_t seed, std::uint64_t index) -> double;

auto generate_input(std::uint64_t seed, std::size_t size, unsigned threads) -> std::vector<double>;
}  // namespace stencil
