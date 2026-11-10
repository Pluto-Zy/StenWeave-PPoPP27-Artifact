from __future__ import annotations


class SASSEmitter:
    def add_barrier(self):
        # = IMPORTANT ==============================================================================
        # The latency of this instruction will affect the correctness of the code.
        # ==========================================================================================
        self.add_code("[B------:R-:W-:-:S06]", "BAR.SYNC.DEFER_BLOCKING 0x0")

    def add_code(self, control_codes: str, code: str, comment=None):
        control_codes = control_codes.strip()
        code = code.strip()

        if code[0] == "@":
            idx = code.index(" ")
            pred = code[:idx]
            code = code[idx + 1 :].strip()
        else:
            pred = ""

        if code[-1] != ";":
            code += ";"

        result = f"    {control_codes} {pred:>4} {code:<60}"

        if comment is not None:
            result = f"    {control_codes} {pred:>4} {code:<60}    // {comment}"
        else:
            result = f"    {control_codes} {pred:>4} {code}"

        self.codes.append(result)

    def add_label(self, label):
        self.codes.append(f"{label}:")

    def add_newline(self):
        self.codes.append("")

    def add_section_title(self, title):
        self.codes.append(f"    //{'=' * 92}//")

        for line in title.splitlines():
            self.codes.append(f"    // {line:<90} //")

        self.codes.append(f"    //{'=' * 92}//")

    def alloc_reg(self, count: int = 1) -> str:
        # Ensure alignment.
        assert self.reg_count % count == 0

        reg = f"R{self.reg_count}"
        self.reg_count += count

        if self.reg_count > 256:
            raise RuntimeError(
                f"Register count exceeds the limit of 256. Current count: {self.reg_count}"
            )

        return reg

    def gmem_load_read_scoreboard(self, _: int) -> int:
        return 5

    def gmem_load_write_scoreboard(self, plane: int) -> int:
        """
        Returns the write scoreboard of the `LDG` instruction that loads the specified plane.
        """
        if plane < 0 or plane > self.precomputed_addr_slot_size:
            plane = self.precomputed_addr_slot_size

        return 0 if plane % 2 == 0 else 3

    def gmem_store_read_scoreboard(self) -> int:
        return 4

    def neighbor_elem_count(self) -> int:
        """
        This function returns the number of FP64 values used to store the neighbor elements loaded
        from shared memory. Note that the number returned by this function is not the number of
        registers, because one FP64 value should be held in two registers.
        """

        # In each group, we need `self.leading_load_count` elements to store the loaded values. If
        # we have more than 1 group, we need to use 2 times the registers to establish group-level
        # scoreboard dependency.
        return (
            self.leading_load_count
            if self.instr_group_count() == 1
            else self.leading_load_count * 2
        )

    @staticmethod
    def wrap(value: int, range: int) -> int:
        value %= range

        if value < 0:
            return value + range
        else:
            return value

    @staticmethod
    def wrap_add(lhs: int, rhs: int, range: int) -> int:
        assert range > 0, "Range must be positive."

        return SASSEmitter.wrap(lhs + rhs, range)

    def emit_operation(self, operation):
        self.add_code(
            operation.to_control_codes(),
            operation.to_sass()
            if operation.pred is None
            else f"@{operation.pred} {operation.to_sass()}",
            operation.comment,
        )
