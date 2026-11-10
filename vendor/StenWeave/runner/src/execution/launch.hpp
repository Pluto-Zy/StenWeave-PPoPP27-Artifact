#pragma once

#include <variant>

namespace stencil::launch {
namespace cuda {
struct Params {
    unsigned height;
};
}  // namespace cuda

namespace sass {
struct Params {
    unsigned row_pitch;
    unsigned plane_stride;
    unsigned cta_stride_x;
    unsigned cta_stride_y;

    unsigned write_begin_x;
    unsigned write_end_x;
    unsigned write_begin_y;
    unsigned write_end_y;

    unsigned pipeline_iterations;
    unsigned pipeline_remainder;

    unsigned store_head;
    unsigned store_tail;
};
}  // namespace sass

using Params = std::variant<cuda::Params, sass::Params>;
}  // namespace stencil::launch
