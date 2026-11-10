"""Rank-independent Box computation strategy; machine order is preserved."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import chain, islice
from typing import Generic, Protocol, TypeVar

from codegen.logical import box_planes
from codegen.sass.dependency import DependencyManager
from codegen.sass.ir import InstrGroup, RegisterPlane
from codegen.state import CodeGenStatus

Index_contra = TypeVar("Index_contra", contravariant=True)


class Plane(Protocol[Index_contra]):
    """Box lifecycle and rank-specific element access, independent of storage layout."""

    in_range: bool
    need_processing: bool

    def to_local(self, index: Index_contra) -> RegisterPlane: ...
    def to_remote(self, index: Index_contra, remote_reg: str) -> RegisterPlane: ...


class BoxComputation:
    def construct_planes(self, status: CodeGenStatus, time_step: int) -> list[Plane]:
        # The index of the target plane, which is the last bottom plane of the current time step.
        logical_planes = box_planes(self.radius, status, time_step)

        def plane_constructor(list_index: int) -> Plane:
            logical = logical_planes[list_index]
            index = logical.index
            target_offset = logical.target_offset
            need_processing = logical.needs_processing
            in_range = logical.in_range

            # The registers of this plane.
            regs = self.regs[time_step][
                type(self).wrap_add(index, status.runtime_offset, self.elements_per_time_step)
            ]

            if time_step == 0 and target_offset == 0:
                slot = self.precomputed_addr_slot(index)
                # Use the precomputed address register if possible.
                if slot is not None:
                    regs = slot

            # Replace `regs` according to `self.replacements`.
            regs = self.replacements.get(self.replacement_key(regs), regs)

            return self.plane_type(
                time_step=time_step,
                index=index,
                in_range=in_range,
                need_processing=need_processing,
                finishes_current_iteration=logical.finishes_current_iteration,
                regs=regs,
            )

        return [plane_constructor(i) for i in range(self.elements_per_time_step)]

    def gen_smem_comp_operations(
        self, planes: list[Plane], dep_manager: DependencyManager
    ) -> list[InstrGroup]:
        """
        This function generates all required shared-memory load/store and computation instructions
        for a given set of planes. It returns the generated instructions as a list of `InstrGroup`.
        Each group is led by at most `self.leading_load_count` shared-memory load instructions,
        along with generated computation instructions.
        """

        previews = self.gen_contribution_plan_preview()
        groups: list[InstrGroup] = [InstrGroup() for _ in previews]

        # In box2d stencil, `planes[-2]` is always the source plane for all contributions.
        src_plane = planes[-2]

        # This list is used to track the last `StencilComputation` that writes to each column. If a
        # column has not been written to yet, its entry will be `None`.
        producers = {}

        if any(plane.need_processing for plane in planes[:-1]):
            # Generate LDS instructions and computation instructions.
            for preview, group in zip(previews, groups, strict=False):
                # Generate shared memory load instructions for each column in `preview.load_columns`
                # and add them to `group.load_instrs`, these instructions load the source values
                # from shared memory.
                group.load_instrs = [
                    self.gen_smem_load(
                        src_plane.to_remote(index=col, remote_reg=self.neighbor_map[col][0]),
                        scoreboard=self.neighbor_map[col][1],
                    )
                    for col in preview.load_columns
                ]

                # Generate computation instructions.
                group.comp_instrs = list(
                    chain.from_iterable(
                        self.preview_to_computations(contribution, producers, planes)
                        for contribution in preview.contributions
                    )
                )

            # Modify the destination plane of the last producer for each local index to be the plane
            # register of `planes[-2]` to ensure that the results can be correctly written back.
            for local_index, producer in producers.items():
                producer.dst_planes[0] = src_plane.to_local(local_index)

            dep_manager.connect_load_groups(groups)

        # Insert the shared-memory store instruction for `planes[-1]`.
        if planes[-1].in_range:
            groups = self.gen_and_distribute_smem_stores(planes[-1], groups)

        return groups


def batched(iterable, n):
    """
    Implements batching of an iterable into tuples of size `n`. The last batch may be smaller than
    `n` if the total number of elements is not divisible by `n`.
    """

    if n < 1:
        raise ValueError("Batch size must be at least one, got {}".format(n))

    iterator = iter(iterable)
    yield from iter(lambda: tuple(islice(iterator, n)), ())


Index = TypeVar("Index")


@dataclass
class ContributionPlanPreview(Generic[Index]):
    """Loads and source/target columns for one Box instruction group."""

    load_columns: list[Index] = field(default_factory=list)
    contributions: list[tuple[Index, Index]] = field(default_factory=list)
