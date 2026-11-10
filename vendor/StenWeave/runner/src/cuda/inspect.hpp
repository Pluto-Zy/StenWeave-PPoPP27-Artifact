#pragma once

#include <nlohmann/json.hpp>

namespace stencil {
auto inspect_device(int ordinal) -> nlohmann::json;
}  // namespace stencil
