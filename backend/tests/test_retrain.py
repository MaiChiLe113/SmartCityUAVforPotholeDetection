from dataclasses import replace

from config import RETRAIN
from ml import retrain


def test_training_history_reads_ultralytics_csv(tmp_path):
    csv = tmp_path / "results.csv"
    csv.write_text("epoch,  metrics/mAP50(B), metrics/mAP50-95(B)\n1,0.40,0.20\n2,0.45,0.22\n")
    rows = retrain.training_history(csv)
    assert rows == [
        {"epoch": 1.0, "metrics/mAP50(B)": 0.40, "metrics/mAP50-95(B)": 0.20},
        {"epoch": 2.0, "metrics/mAP50(B)": 0.45, "metrics/mAP50-95(B)": 0.22},
    ]


def test_next_version():
    assert retrain.next_version("v1") == "v2"
    assert retrain.next_version("v9") == "v10"
    assert retrain.next_version("champion") == "v2"


def test_seeded_sample_is_stable():
    items = [f"img_{i}" for i in range(100)]
    a = retrain.seeded_sample(items, 10, 42)
    assert a == retrain.seeded_sample(list(reversed(items)), 10, 42)
    assert len(a) == 10


def test_current_version_defaults_to_v1(tmp_path):
    cfg = replace(RETRAIN, metadata_dir=tmp_path)
    assert retrain.current_version(cfg) == "v1"
    (tmp_path / "model_version.txt").write_text("v3\n")
    assert retrain.current_version(cfg) == "v3"


def test_cpu_split_keeps_new_images_and_samples_old(tmp_path):
    data = tmp_path / "data"
    rdir = data / "retrain"
    rdir.mkdir(parents=True)
    old = [f"./../prototype/images/train/rdd_Japan_Japan_{i:06d}.jpg" for i in range(50)]
    new = [f"./../new/batch_001/images/rdd_Czech_Czech_{i:06d}.jpg" for i in range(5)]
    (rdir / "train.txt").write_text("\n".join(old + new) + "\n")
    (rdir / "val.txt").write_text("\n".join(old[:20] + new[:2]) + "\n")
    cfg = replace(RETRAIN, retrain_dir=rdir, cpu_dir=data / "retrain_cpu", cpu_replay=10, cpu_val=8)
    yaml, needed = retrain.cpu_split(cfg)
    train = (cfg.cpu_dir / "train.txt").read_text().splitlines()
    val = (cfg.cpu_dir / "val.txt").read_text().splitlines()
    assert len(train) == 15 and sum("/new/" in t for t in train) == 5
    assert len(val) == 8 and sum("/new/" in v for v in val) == 2
    assert len(needed) == 23 and yaml.exists()
