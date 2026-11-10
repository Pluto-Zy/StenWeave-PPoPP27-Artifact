from __future__ import annotations

from dataclasses import dataclass, field

from codegen.sass.ir import InstrGroup, RegisterPlane, SASSOperation


@dataclass
class Dependency:
    producer: SASSOperation
    consumer: SASSOperation
    required_stall: int


@dataclass
class DependencyManager:
    # Records the mapping from each `RegisterPlane` to the last computation instruction that writes to
    # its register.
    plane_producer: dict[RegisterPlane, SASSOperation] = field(default_factory=dict)
    dependencies: list[Dependency] = field(default_factory=list)

    def add_dependencies(
        self, operations: list[SASSOperation], stall_dict: dict[tuple[type, type], int]
    ):
        for operation in operations:
            self.add_dependency_for(operation, stall_dict)

    def connect_load_groups(self, groups: list[InstrGroup]) -> None:
        """Connect existing LDS groups without changing their machine order."""
        for index, group in enumerate(groups):
            if index + 1 != len(groups):
                next_group = groups[index + 1]
                common_size = min(len(group.load_instrs), len(next_group.load_instrs))

                for producer, consumer in zip(
                    group.load_instrs[:common_size],
                    next_group.load_instrs[:common_size],
                    strict=False,
                ):
                    write_scoreboard = producer.write_scoreboard
                    assert write_scoreboard is not None

                    consumer.wait_scoreboard.add(write_scoreboard)
                    self.add_dependency(producer, consumer, stall=2)
            else:
                next_group = group
                common_size = 0

            for load in group.load_instrs[common_size:]:
                write_scoreboard = load.write_scoreboard
                assert write_scoreboard is not None

                for operation in next_group.comp_instrs:
                    if any(
                        load.dst_planes[0].reg() == source.reg() for source in operation.src_planes
                    ):
                        operation.wait_scoreboard.add(write_scoreboard)
                        break

    def add_dependency_for(
        self, operation: SASSOperation, stall_dict: dict[tuple[type, type], int]
    ):
        """
        This function establishes dependencies from the last instruction that writes to each
        `RegisterPlane` in `operation.src_planes` to `operation`, with enough stall cycles of
        separation. In other words, `operation` must be executed after all instructions that write
        to its source operands.

        `stall_dict` is a dictionary that maps a tuple of two instruction types to the required
        stall cycles between them. The first type in the tuple is the producer instruction type, and
        the second type is the consumer instruction type.

        The function also records `operation` as the last writer for each `RegisterPlane` in its
        `dst_planes`, so that subsequent computation instructions can establish dependencies.
        """

        for plane in operation.src_planes:
            producer = self.plane_producer.get(plane)
            if producer is not None:
                dependency = Dependency(
                    producer=producer,
                    consumer=operation,
                    required_stall=stall_dict[type(producer), type(operation)],
                )
                self.dependencies.append(dependency)

        for plane in operation.dst_planes:
            self.plane_producer[plane] = operation

    def add_dependency(self, producer: SASSOperation, consumer: SASSOperation, stall: int):
        self.dependencies.append(
            Dependency(producer=producer, consumer=consumer, required_stall=stall)
        )

    def compute_min_stalls(self, operations: list[SASSOperation], per_op_min_stall: int = 1):
        """
        Computes the minimum stall cycles for each operation that satisfies the dependencies.
        """

        graph = [[] for _ in range(len(operations) + 1)]
        for i, op in enumerate(operations):
            min_stall = op.stall if op.stall != 0 else per_op_min_stall
            graph[i].append((i + 1, min_stall))

        for dep in self.dependencies:
            # Find the index of the producer and consumer operations.
            producer_index = operations.index(dep.producer)
            consumer_index = operations.index(dep.consumer)
            assert producer_index < consumer_index

            graph[producer_index].append((consumer_index, dep.required_stall))

        y = [0] * (len(operations) + 1)
        for u in range(len(operations) + 1):
            for v, w in graph[u]:
                y[v] = max(y[v], y[u] + w)

        for id, op in enumerate(operations):
            op.stall = y[id + 1] - y[id]
