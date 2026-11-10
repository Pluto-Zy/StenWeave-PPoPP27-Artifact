"""Names and cumulative features for the four breakdown stages."""

STAGE_NAMES = {
    1: "baseline",
    2: "segmented SMEM",
    3: "Comm-Centric SMEM Scheduling",
    4: "layered point update",
}


def features(stage: int, shape: str, radius: int) -> dict[str, bool]:
    if stage not in STAGE_NAMES:
        raise ValueError("breakdown stage must be 1, 2, 3, or 4")

    result = {
        "layered_point_update": stage >= 4,
        "comm_centric_smem_scheduling": stage >= 3,
        "segmented_smem": stage >= 2,
    }

    if shape == "box" and radius == 1 and stage == 3:
        result["gather_row_contributions"] = True

    if shape == "star" and radius == 1 and stage < 4:
        result["gather_update"] = True

    return result
