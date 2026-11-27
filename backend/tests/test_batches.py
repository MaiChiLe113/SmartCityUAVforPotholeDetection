from dataclasses import replace

from config import DATASET_V1
from ml import batches


def test_split_into_batches_covers_every_stem_once():
    stems = [f"s{i}" for i in range(10)]
    parts = batches.split_into_batches(stems, 3, seed=42)
    assert sorted(s for part in parts for s in part) == sorted(stems)
    assert sorted(len(p) for p in parts) == [3, 3, 4]
    assert parts == batches.split_into_batches(list(reversed(stems)), 3, seed=42)


def test_unified_numbers_reads_both_sides(tmp_path):
    for side, name in (("train", "rdd_Czech_Czech_000010.jpg"), ("val", "rdd_Czech_Czech_000020.jpg")):
        folder = tmp_path / "images" / side
        folder.mkdir(parents=True)
        (folder / name).write_bytes(b"x")
    (tmp_path / "images" / "train" / "rdd_Japan_Japan_000001.jpg").write_bytes(b"x")
    assert batches.unified_numbers("Czech", replace(DATASET_V1, unified_dir=tmp_path)) == {"000010", "000020"}
