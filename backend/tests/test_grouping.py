from collections import Counter

import pytest

from ml import grouping

RDD_DRONE = "rdd_China_Drone_China_Drone_000745"
RDV_DRONE_A = "rdv_China_Drone_000745_jpg.rf.aa11"
RDV_DRONE_B = "rdv_China_Drone_000745_jpg.rf.bb22"
RDV_A = "rdv_285_png_jpg.rf.cc33"
RDV_B = "rdv_285_png_jpg.rf.dd44"
GROUND_A = "rdd_United_States_United_States_003917"
GROUND_B = "rdd_United_States_United_States_001623"
SHARE = {"ground": 0.15, "aerial": 0.20}


def uav(prefix: str, number: str, corner: str = "top_left") -> str:
    return f"uav_{prefix}_{number}_{corner}"


def group_containing(groups: dict[str, list[str]], stem: str) -> list[str]:
    return next(members for members in groups.values() if stem in members)


def test_rdd_drone_photo_and_its_rdv_copies_form_one_group():
    groups = grouping.group_images([RDD_DRONE, RDV_DRONE_A, RDV_DRONE_B, GROUND_A])
    assert group_containing(groups, RDD_DRONE) == [RDD_DRONE, RDV_DRONE_A, RDV_DRONE_B]
    assert group_containing(groups, GROUND_A) == [GROUND_A]


def test_rdv_exports_of_one_base_form_one_group():
    assert len(grouping.group_images([RDV_A, RDV_B])) == 1


def test_uav_flips_and_tiles_of_one_frame_form_one_group():
    stems = [uav("lr", "7"), uav("tb", "7", "bottom_right"), uav("r180", "7"), uav("lr", "8")]
    groups = grouping.group_images(stems)
    assert len(groups) == 2
    assert uav("lr", "8") not in group_containing(groups, uav("lr", "7"))


def test_identical_bytes_merge_groups():
    assert len(grouping.group_images([GROUND_A, GROUND_B])) == 2
    assert len(grouping.group_images([GROUND_A, GROUND_B], {GROUND_A: "same", GROUND_B: "same"})) == 1


def test_unknown_name_is_rejected():
    with pytest.raises(ValueError):
        grouping.stratum_of("mystery_0001")


def test_views():
    assert grouping.view_of(RDD_DRONE) == "aerial"
    assert grouping.view_of(RDV_A) == "aerial"
    assert grouping.view_of(uav("lr", "1")) == "aerial"
    assert grouping.view_of(GROUND_A) == "ground"


def test_validation_prefers_the_original_over_copies():
    assert grouping.val_representatives([RDD_DRONE, RDV_DRONE_A, RDV_DRONE_B]) == [RDD_DRONE]


def test_validation_keeps_one_flip_version_of_every_tile():
    lr = [uav("lr", "7", "top_left"), uav("lr", "7", "top_right")]
    tb = [uav("tb", "7", "top_left"), uav("tb", "7", "top_right")]
    assert grouping.val_representatives(sorted(lr + tb)) == lr


def synthetic_dataset() -> tuple[dict[str, list[str]], dict[str, Counter]]:
    stems = []
    for i in range(100):
        stems += [f"rdd_China_Drone_China_Drone_{i:06d}", f"rdv_China_Drone_{i:06d}_jpg.rf.aa"]
        stems += [uav(p, str(i)) for p in ("lr", "tb", "r180")]
        stems.append(f"rdd_Japan_Japan_{i:06d}")
    boxes = {s: Counter({int(s[-1], 36) % 4: 1}) for s in stems}
    return grouping.group_images(stems), boxes


def test_split_keeps_groups_whole_and_uses_the_configured_shares():
    groups, boxes = synthetic_dataset()
    split = grouping.split_groups(groups, boxes, SHARE, seed=42)
    grouping.assert_no_leak(split)
    val_groups = {split.group_of[s] for s in split.val}
    assert sum(g.startswith("rdd_Japan") for g in val_groups) == 15
    assert sum(g.startswith("rdd_China_Drone") for g in val_groups) == 20
    assert sum(g.startswith("uav_") for g in val_groups) == 20
    assert len(split.train) + len(split.val) + len(split.excluded) == sum(len(m) for m in groups.values())


def test_split_is_repeatable_and_depends_on_the_seed():
    groups, boxes = synthetic_dataset()
    first = grouping.split_groups(groups, boxes, SHARE, seed=1)
    assert first == grouping.split_groups(groups, boxes, SHARE, seed=1)
    assert first.val != grouping.split_groups(groups, boxes, SHARE, seed=2).val


def test_leak_check_catches_a_group_on_both_sides():
    split = grouping.GroupedSplit([RDD_DRONE], [RDV_DRONE_A], [], {RDD_DRONE: RDD_DRONE, RDV_DRONE_A: RDD_DRONE})
    with pytest.raises(ValueError, match="groups cross"):
        grouping.assert_no_leak(split)


def test_leak_check_catches_identical_files_on_both_sides():
    split = grouping.GroupedSplit([GROUND_A], [GROUND_B], [], {GROUND_A: GROUND_A, GROUND_B: GROUND_B})
    with pytest.raises(ValueError, match="identical files"):
        grouping.assert_no_leak(split, {GROUND_A: "same", GROUND_B: "same"})
