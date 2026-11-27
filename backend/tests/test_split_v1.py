from collections import Counter

import pytest

from ml import split_v1

SHARES = {"val": {"ground": 0.15, "aerial": 0.15}, "test": {"ground": 0.15, "aerial": 0.15}}


def ground(i: int) -> str:
    return f"rdd_Japan_Japan_{i:06d}"


def tile(frame: int, corner: str) -> str:
    return f"uavpdd_{frame:04d}_{corner}"


def test_copy_and_frame_links_both_group_but_only_copies_merge_photos():
    stems = ["rdd_China_Drone_China_Drone_000001", "rdv_China_Drone_000001_jpg.rf.aa", tile(1, "a"), tile(1, "b")]
    groups, photo_of = split_v1.build_groups(stems, [(stems[0], stems[1])], [(stems[2], stems[3])])
    assert len(groups) == 2
    assert photo_of[stems[0]] == photo_of[stems[1]]
    assert photo_of[stems[2]] != photo_of[stems[3]]
    assert split_v1.representatives(sorted(stems[:2]), photo_of) == [stems[0]]
    assert split_v1.representatives(stems[2:], photo_of) == stems[2:]


def dataset(n: int = 200):
    stems = [ground(i) for i in range(n)] + [tile(i, c) for i in range(n) for c in "ab"]
    frames = [(tile(i, "a"), tile(i, "b")) for i in range(n)]
    groups, photo_of = split_v1.build_groups(stems, [], frames)
    boxes = {s: Counter({i % 4: 1}) for i, s in enumerate(stems)}
    return groups, photo_of, boxes


def test_three_way_split_uses_the_shares_and_keeps_groups_whole():
    groups, photo_of, boxes = dataset()
    split = split_v1.split_three_way(groups, photo_of, boxes, set(), SHARES, seed=7)
    split_v1.assert_disjoint(split)
    per_side = Counter(split.group_side.values())
    assert per_side == {"train": 280, "val": 60, "test": 60}
    assert split.excluded == {}


def test_negatives_never_reach_train():
    groups, photo_of, boxes = dataset()
    negatives = {ground(i) for i in range(0, 200, 10)}
    for stem in negatives:
        boxes[stem] = Counter()
    split = split_v1.split_three_way(groups, photo_of, boxes, negatives, SHARES, seed=7)
    assert not negatives & set(split.sides["train"])
    assert negatives <= set(split.sides["val"]) | set(split.sides["test"])


def test_split_is_repeatable():
    groups, photo_of, boxes = dataset()
    a = split_v1.split_three_way(groups, photo_of, boxes, set(), SHARES, seed=3)
    assert a == split_v1.split_three_way(groups, photo_of, boxes, set(), SHARES, seed=3)


def test_disjoint_check_catches_identical_bytes():
    split = split_v1.SplitV1(
        {"train": ["a"], "val": ["b"], "test": []}, {}, {"a": "a", "b": "b"}, {"a": "train", "b": "val"}
    )
    with pytest.raises(ValueError, match="identical"):
        split_v1.assert_disjoint(split, {"a": "x", "b": "x"})


def test_lock_detects_a_changed_label(tmp_path):
    image, label = tmp_path / "x.jpg", tmp_path / "x.txt"
    image.write_bytes(b"img")
    label.write_text("0 0.5 0.5 0.1 0.1\n")
    lock = tmp_path / "lock.json"
    split_v1.write_test_lock([image], lambda p: label, lock)
    split_v1.verify_test_lock([image], lambda p: label, lock)
    label.write_text("1 0.5 0.5 0.1 0.1\n")
    with pytest.raises(ValueError):
        split_v1.verify_test_lock([image], lambda p: label, lock)


def test_near_duplicates_found_through_any_pose():
    pytest.importorskip("numpy")
    hashes = {"a": [0b1011] + [7] * 7, "b": [0xFFFF] + [0b1010] + [9] * 6, "c": [2**63] * 8}
    pairs = split_v1.near_duplicate_pairs(hashes, max_distance=1)
    assert pairs == [("a", "b", 1)]
