from dataclasses import replace

import pytest

from config import DATASET_V1, RETRAIN
from ml import ingest


def test_label_errors_accepts_valid_and_empty_labels():
    assert ingest.label_errors("0 0.5 0.5 0.2 0.1\n3 0.1 0.9 0.05 0.05\n", 4) == []
    assert ingest.label_errors("", 4) == []


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("4 0.5 0.5 0.2 0.2", "class 4"),
        ("0 0.5 0.5 0.2", "expected 5"),
        ("0 1.5 0.5 0.2 0.2", "outside [0, 1]"),
        ("0 0.5 0.5 0 0.2", "empty box"),
        ("a 0.5 0.5 0.2 0.2", "not numeric"),
    ],
)
def test_label_errors_reports_each_problem(text, fragment):
    errors = ingest.label_errors(text, 4)
    assert len(errors) == 1 and fragment in errors[0]


def test_goes_to_val_is_stable_and_near_share():
    groups = [f"g{i}" for i in range(4000)]
    first = [ingest.goes_to_val(g, 0.15, 42) for g in groups]
    assert first == [ingest.goes_to_val(g, 0.15, 42) for g in groups]
    assert 0.12 < sum(first) / len(first) < 0.18


def test_group_new_images_joins_links_and_near_duplicates():
    stems = ["rdd_China_Drone_China_Drone_000010", "rdv_China_Drone_000010_jpg.rf.ab12", "a", "b", "c"]
    groups = ingest.group_new_images(stems, [("a", "b")])
    assert groups[stems[0]] == groups[stems[1]]
    assert groups["a"] == groups["b"] != groups["c"]


def test_cross_candidates_finds_close_hashes():
    new = {"n1": [0b1111] + [2**63] * 7, "n2": [2**40] * 8}
    ref = {"r1": [0b1110] + [0] * 7}
    assert ingest.cross_candidates(new, ref, max_distance=2) == [("n1", "r1", 1)]


def _write_image(path, seed):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), rng.integers(0, 255, (96, 128, 3), dtype=np.uint8))


def _setup(tmp_path):
    """A tiny dataset v1 (one val, one test image) and one new batch."""
    v1_dir = tmp_path / "data" / "interim_v1"
    images = tmp_path / "data" / "pool" / "images"
    _write_image(images / "rdd_Japan_Japan_000001.jpg", 1)  # val
    _write_image(images / "rdd_Japan_Japan_000002.jpg", 2)  # test
    _write_image(images / "rdd_Japan_Japan_000003.jpg", 3)  # train
    v1_dir.mkdir(parents=True)
    (v1_dir / "train.txt").write_text("./../pool/images/rdd_Japan_Japan_000003.jpg\n")
    (v1_dir / "val.txt").write_text("./../pool/images/rdd_Japan_Japan_000001.jpg\n")
    (v1_dir / "test.txt").write_text("./../pool/images/rdd_Japan_Japan_000002.jpg\n")

    batch = tmp_path / "data" / "new" / "batch_001"
    for i in range(10, 16):
        _write_image(batch / "images" / f"rdd_Czech_Czech_{i:06d}.jpg", i)
        (batch / "labels").mkdir(parents=True, exist_ok=True)
        (batch / "labels" / f"rdd_Czech_Czech_{i:06d}.txt").write_text("3 0.5 0.5 0.2 0.2\n")
    # a byte copy of the test image and a bad label
    (batch / "images" / "rdd_Czech_Czech_000099.jpg").write_bytes((images / "rdd_Japan_Japan_000002.jpg").read_bytes())
    (batch / "labels" / "rdd_Czech_Czech_000099.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    (batch / "labels" / "rdd_Czech_Czech_000015.txt").write_text("7 0.5 0.5 0.2 0.2\n")

    cfg = replace(
        RETRAIN,
        new_data_dir=tmp_path / "data" / "new",
        retrain_dir=tmp_path / "data" / "retrain",
        metadata_dir=tmp_path / "artifacts" / "metadata",
        monitoring_dir=tmp_path / "monitoring",
    )
    v1 = replace(DATASET_V1, output_dir=v1_dir)
    return cfg, v1


def test_ingest_rejects_leaks_and_bad_labels_and_keeps_test_frozen(tmp_path):
    cfg, v1 = _setup(tmp_path)
    report = ingest.ingest(cfg, v1)
    reasons = {r["stem"]: r["reason"] for r in report.rejected}
    assert reasons["rdd_Czech_Czech_000099"].startswith("copy_of_test")
    assert reasons["rdd_Czech_Czech_000015"].startswith("bad_label")
    assert len(report.train) + len(report.val) == 5

    yaml = ingest.write_retrain_split(report, cfg, v1)
    text = yaml.read_text()
    assert "test: ../interim_v1/test.txt" in text
    train = (cfg.retrain_dir / "train.txt").read_text().splitlines()
    assert train[0] == "./../pool/images/rdd_Japan_Japan_000003.jpg"
    assert all(line.startswith("./") for line in train)
    assert (v1.output_dir / "test.txt").read_text() == "./../pool/images/rdd_Japan_Japan_000002.jpg\n"


def test_pending_batches_skips_batches_the_champion_was_trained_on(tmp_path):
    cfg, _ = _setup(tmp_path)
    assert [b.name for b in ingest.pending_batches(cfg)] == ["batch_001"]
    cfg.metadata_dir.mkdir(parents=True)
    (cfg.metadata_dir / "batches.json").write_text('{"batch_001": {"images": 7}}')
    assert ingest.pending_batches(cfg) == []
