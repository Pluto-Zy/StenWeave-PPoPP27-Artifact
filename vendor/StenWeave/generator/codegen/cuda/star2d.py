"""Scalar CUDA Star lowering of the shared logical plane lifecycle."""

from codegen.logical import (
    StarTopology,
    StencilComputationSource,
    plane_in_range,
    star_coefficient_index_2d,
    star_planes,
)
from codegen.state import CodeGenStatus, Phase


class CodeGen:
    def __init__(self, radius: int, time_block_size: int, overlap_size: int):
        self.radius = radius
        self.time_block_size = time_block_size
        self.overlap_size = overlap_size
        self.topology = StarTopology(radius, time_block_size)

        self.codes = []
        self.pending = {}
        self.chunk = time_block_size + (time_block_size % 2 == 0)

    def generate_reg_declarations(self) -> str:
        return ""

    def reg(self, time_step: int, plane: int, runtime_offset: int) -> str:
        return (
            f"value_{time_step}_{(plane + runtime_offset) % self.topology.elements_per_time_step}"
        )

    def stencil_pipeline(self, status: CodeGenStatus, parity: int) -> None:
        add = self.codes.append

        for destination, temporary in self.pending.pop(status, {}).items():
            add(f"{destination} = {temporary};")

        r = self.radius
        add("{")

        for t in range(self.time_block_size):
            planes = star_planes(r, status, t)
            target = planes[-1]
            current = self.reg(t, target.index, status.runtime_offset)
            add(f"double const incoming_{t} = {current};")

            horizontal = planes[r - 1]
            if StencilComputationSource.OtherColumn in horizontal.computation_sources:
                for distance in range(1, r + 1):
                    for offset in (distance, -distance):
                        name = f"neighbor_{t}_{offset + r}"
                        add(
                            f"double const {name} = exchange_{1 - parity}[{offset * self.chunk + t}];"
                        )

            if target.in_range:
                add(f"exchange_{parity}[{t}] = incoming_{t};")

            for dst in planes[:-1]:
                if StencilComputationSource.SameColumn not in dst.computation_sources:
                    continue

                destination = self.reg(t, dst.index, status.runtime_offset)
                coeff = star_coefficient_index_2d(0, target.index - dst.index - r)

                if target.index == 0:
                    add(f"{destination} = incoming_{t} * neighbor_coeffs[{coeff}];")
                else:
                    add(
                        f"{destination} = fma(incoming_{t}, neighbor_coeffs[{coeff}], {destination});"
                    )

            # CUDA keeps the published input and its first accumulator separate.
            # Initialize that accumulator now, before the next barrier, rather
            # than putting its multiply on the next row's completion path.
            if target.in_range and target.meaningful:
                coeff = star_coefficient_index_2d(0, -r)
                add(f"{current} = incoming_{t} * neighbor_coeffs[{coeff}];")

            horizontal = planes[r - 1]
            if StencilComputationSource.OtherColumn in horizontal.computation_sources:
                dst = self.reg(t, horizontal.index, status.runtime_offset)

                for distance in range(1, r + 1):
                    for offset in (distance, -distance):
                        coeff = star_coefficient_index_2d(offset, 0)
                        add(
                            f"{dst} = fma(neighbor_{t}_{offset + r}, neighbor_coeffs[{coeff}], {dst});"
                        )

            for index, plane in enumerate(planes):
                if not plane.finishes_current_iteration:
                    continue

                value = self.reg(t, plane.index, status.runtime_offset)

                if t + 1 < self.time_block_size:
                    next_value = self.reg(t + 1, plane.index + r, status.runtime_offset)

                    if index == 0:
                        # Publish through the final producer, releasing the old
                        # accumulator without overwriting an aliased live value.
                        for operation_index in range(len(self.codes) - 1, -1, -1):
                            statement = self.codes[operation_index]
                            if statement.startswith(f"{value} ="):
                                self.codes[operation_index] = next_value + statement[len(value) :]
                                break
                        else:
                            raise RuntimeError("Completed plane has no producer")
                    else:
                        activation = CodeGenStatus(
                            status.phase, status.plane + index, status.runtime_offset
                        )
                        temporary = f"deferred_{t}_{(plane.index + status.runtime_offset) % self.topology.elements_per_time_step}"
                        add(f"{temporary} = {value};")
                        self.pending.setdefault(activation, {})[next_value] = temporary
                else:
                    add(f"if (owns_output) *output_cursor = {value}; output_cursor += width;")

            if t == 0 and plane_in_range(status.plane + 1, status.phase):
                add(
                    f"{self.reg(0, status.plane + 1, status.runtime_offset)} = *input_cursor; input_cursor += width;"
                )

        add("__syncthreads();")
        add("}")

    def generate(self):
        add = self.codes.append
        topology = self.topology

        for t in range(self.time_block_size):
            for p in range(topology.elements_per_time_step):
                add(f"double value_{t}_{p} = 0;")

        add(
            f"unsigned const width = gridDim.x * blockDim.x - {self.overlap_size} * (gridDim.x - 1);"
        )
        add(
            f"unsigned const column = (blockDim.x - {self.overlap_size}) * blockIdx.x + threadIdx.x;"
        )

        add("double* input_cursor = input + column + width;")
        add("double* output_cursor = output + column;")
        add(f"{self.reg(0, 0, 0)} = input[column];")

        add(f"double* exchange_0 = shared_mem + (threadIdx.x + {self.radius}) * {self.chunk};")
        add(
            f"double* exchange_1 = shared_mem + (blockDim.x + {2 * self.radius} + threadIdx.x) * {self.chunk};"
        )

        half = self.overlap_size // 2
        add(
            f"bool const owns_output = (threadIdx.x >= {half} || blockIdx.x == 0) && (threadIdx.x < blockDim.x - {half} || blockIdx.x + 1 == gridDim.x);"
        )

        add(f"unsigned const rows = height - {topology.head_end - topology.tail_start};")
        add(f"unsigned const iterations = rows / {topology.planes_per_iteration};")
        add(f"unsigned const remain = rows % {topology.planes_per_iteration};")

        for plane in range(topology.head_end):
            self.stencil_pipeline(CodeGenStatus(Phase.Head, plane), plane % 2)

        start = topology.head_end
        add("for (unsigned iteration = 0; iteration < iterations; ++iteration) {")

        for i in range(topology.planes_per_iteration):
            self.stencil_pipeline(CodeGenStatus(Phase.Pipeline, start + i), (start + i) % 2)

        add("}")

        add("switch (remain) {")
        for remain in range(topology.planes_per_iteration):
            self.pending.clear()
            add(f"case {remain}: {{")
            for t in range(self.time_block_size - 1):
                for p in range(topology.elements_per_time_step):
                    add(f"double deferred_{t}_{p};")

            for i in range(remain):
                self.stencil_pipeline(CodeGenStatus(Phase.Pipeline, start + i), (start + i) % 2)

            for i, plane in enumerate(
                range(topology.tail_start, topology.head_end + topology.tail_start)
            ):
                self.stencil_pipeline(
                    CodeGenStatus(Phase.Tail, plane, start + remain - topology.tail_start),
                    (start + remain + i) % 2,
                )

            add("return; }")

        add("}")
