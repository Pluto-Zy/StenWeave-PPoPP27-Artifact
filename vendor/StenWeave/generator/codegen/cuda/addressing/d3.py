"""Historical 3D CUDA volume addressing and fixed-overlap strip prologue."""

from codegen.cuda.pipeline import VolumePipeline


class VolumeAddressing(VolumePipeline):
    def __init__(self, radius: int, time_block_size: int, overlap_size: int):
        assert 0 < radius < 4
        assert overlap_size % 2 == 0

        self.topology = self.topology_type(radius, time_block_size)
        self.codes = []
        self.radius = radius
        self.time_block_size = time_block_size
        self.overlap_size = overlap_size
        self.elements_per_time_step = self.topology.elements_per_time_step
        self.smem_chunk_size = self.time_block_size + (1 if self.time_block_size % 2 == 0 else 0)
        self.planes_per_iteration = self.topology.planes_per_iteration
        self.precomputed_addr_slot_size = self.topology.head_end - 1 - self.address_slot_adjustment

        self.replacements = {}
        self.candidate_replacements = {}

        self.global_mem_stride = "global_mem_stride"
        self.input = "input"
        self.output = "output"

        self.shared_mem_reads = [f"shared_mem_read_{i}" for i in range(2 * self.radius + 1)]
        self.shared_mem_writes = [f"shared_mem_write_{i}" for i in range(2 * self.radius + 1)]

        self.neighbors = [f"neighbor_{i}" for i in range(self.neighbor_count)]
        self.write_result = "write_result"
        self.pipeline_phase_iterations = "pipeline_phase_iterations"
        self.remain_rows = "remain_rows"

        self.regs = tuple(
            tuple(f"reg_{time_step_start}_{i}" for i in range(self.elements_per_time_step))
            for time_step_start in range(self.time_block_size)
        )
        self.code_indent = 4

    def gen_prologue(self):
        self.add_section_title("Prologue")
        codes = []
        codes.append(
            f"{self.global_mem_stride} = ((gridDim.x * (blockDim.x - {self.overlap_size}) + {self.overlap_size}) * (gridDim.y * (blockDim.y - {self.overlap_size}) + {self.overlap_size}));"
        )
        codes.extend(
            f"\n            unsigned const plane_x_length = {self.plane_width_expression()};\n            unsigned const ptr_offset = plane_x_length * ((blockDim.y - {self.overlap_size}) * blockIdx.y + threadIdx.y) + (blockDim.x - {self.overlap_size}) * blockIdx.x + threadIdx.x;\n            {self.input} = {self.input} + ptr_offset;\n            {self.output} = {self.output} + ptr_offset;\n            ".splitlines()
        )

        codes.append(f"{self.regs[0][0]} = *{self.input};")
        codes.extend(
            f"{self.precomputed_addr_slot(row)} = bit_cast<double>({self.input} + {self.global_mem_stride} * {row});"
            for row in range(1, self.precomputed_addr_slot_size)
        )
        codes.append(f"{self.regs[0][1]} = *bit_cast<double *>({self.precomputed_addr_slot(1)});")
        codes.extend(
            f"\n            unsigned const pipeline_rows = height - {self.topology.head_end - self.topology.tail_start};\n            {self.pipeline_phase_iterations} = pipeline_rows / {self.planes_per_iteration};\n            {self.remain_rows} = pipeline_rows % {self.planes_per_iteration};\n            ".splitlines()
        )
        codes.extend(
            f"\n            unsigned const offset_base = (threadIdx.y * (blockDim.x + {self.radius}) + threadIdx.x + {self.radius}) * {self.smem_chunk_size};\n            {self.shared_mem_reads[0]} = shared_mem + offset_base;\n            int const row_offset = (blockDim.x + {self.radius}) * {self.smem_chunk_size};\n            unsigned const max_smem_read_size = {self.shared_plane_extent()};\n            ".splitlines()
        )

        for i in range(1, self.radius + 1):
            codes.append(
                f"{self.shared_mem_reads[i]} = shared_mem + min(offset_base + {i} * row_offset, max_smem_read_size);"
            )

        for i in range(-1, -self.radius - 1, -1):
            codes.append(
                f"{self.shared_mem_reads[i]} = shared_mem + min(offset_base - {-i} * row_offset, max_smem_read_size);"
            )

        for i in range(2 * self.radius + 1):
            codes.append(
                f"{self.shared_mem_writes[i]} = {self.shared_mem_reads[i]} + max_smem_read_size{self.shared_plane_gap()};"
            )

        codes.append(
            f"{self.write_result} = (threadIdx.x >= {self.overlap_size // 2} || blockIdx.x == 0) && (threadIdx.x < blockDim.x - {self.overlap_size // 2} || blockIdx.x == gridDim.x - 1) && (threadIdx.y >= {self.overlap_size // 2} || blockIdx.y == 0) && (threadIdx.y < blockDim.y - {self.overlap_size // 2} || blockIdx.y == gridDim.y - 1);"
        )

        for code in codes:
            self.add_code(code)

    def plane_width_expression(self):
        expression = f"gridDim.x * (blockDim.x - {self.overlap_size}) + {self.overlap_size}"
        return f"({expression})" if self.full_cross_section else expression

    def shared_plane_extent(self):
        if self.full_cross_section:
            return f"((blockDim.x + {self.radius}) * blockDim.y + {self.radius}) * {self.smem_chunk_size}"

        return "row_offset * blockDim.y"

    def shared_plane_gap(self):
        return f" + {self.smem_chunk_size}" if self.full_cross_section else ""

    @property
    def publication_buffers(self):
        return self.shared_mem_reads, self.shared_mem_writes

    @publication_buffers.setter
    def publication_buffers(self, buffers):
        self.shared_mem_reads, self.shared_mem_writes = buffers
