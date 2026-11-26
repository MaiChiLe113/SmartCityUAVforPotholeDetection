from ml.thresholds import best_conf_per_class


def test_threshold_is_the_f1_peak_of_each_present_class():
    px = [0.1, 0.2, 0.3, 0.4]
    f1 = [[0.2, 0.6, 0.5, 0.1], [0.1, 0.2, 0.3, 0.9]]
    p = [[0.1, 0.7, 0.8, 0.9], [0.1, 0.2, 0.3, 0.95]]
    r = [[0.9, 0.5, 0.3, 0.05], [0.9, 0.8, 0.6, 0.85]]
    out = best_conf_per_class(px, f1, p, r, class_index=[0, 3], class_names=["D00", "D10", "D20", "D40"])
    assert out["D00"] == {"conf": 0.2, "f1": 0.6, "precision": 0.7, "recall": 0.5}
    assert out["D40"]["conf"] == 0.4
    assert set(out) == {"D00", "D40"}
