#pragma once

#include <filesystem>
#include <string>

namespace stencil {
auto sha256_file(std::filesystem::path const &path) -> std::string;
}  // namespace stencil
