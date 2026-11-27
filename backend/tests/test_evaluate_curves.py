from ml.evaluate import at_confidence


def test_curve_value_is_read_at_the_nearest_confidence():
    px = [0.0, 0.25, 0.5, 0.75]
    assert at_confidence(px, [1, 2, 3, 4], 0.3) == 2.0
    assert at_confidence(px, [1, 2, 3, 4], 0.9) == 4.0
