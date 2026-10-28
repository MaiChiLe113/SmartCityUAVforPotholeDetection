import config


def test_split_sums_to_one():
    assert abs(sum(config.DATA.split) - 1.0) < 1e-9


def test_four_classes_with_names():
    assert config.DATA.classes == ("D00", "D10", "D20", "D40")
    assert len(config.DATA.class_names) == len(config.DATA.classes)


def test_thresholds_cover_every_class():
    assert set(config.INFER.conf_thresholds) == set(config.DATA.classes)


def test_three_candidates_same_imgsz():
    assert len(config.TRAIN.candidates) == 3
    assert config.TRAIN.imgsz == config.INFER.imgsz == config.EXPORT.imgsz == 640
