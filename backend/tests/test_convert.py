from pathlib import Path

import pytest

from ml import convert

CLASS_MAP = {"LC": 0, "TC": 1, "AC": 2, "PH": 3}


def write_voc(path: Path, objects: list[tuple[str, int, int, int, int]], size=(200, 100)) -> Path:
    body = "".join(
        f"<object><name>{n}</name><bndbox><xmin>{a}</xmin><ymin>{b}</ymin><xmax>{c}</xmax><ymax>{d}</ymax>"
        "</bndbox></object>"
        for n, a, b, c, d in objects
    )
    path.write_text(
        f"<annotation><filename>{path.stem}.jpg</filename><size><width>{size[0]}</width>"
        f"<height>{size[1]}</height><depth>3</depth></size>{body}</annotation>"
    )
    return path


def test_mapped_boxes_become_normalised_yolo_lines(tmp_path):
    image = convert.parse_voc(write_voc(tmp_path / "a.xml", [("PH", 50, 25, 150, 75), ("OC", 0, 0, 10, 10)]))
    lines, dropped = convert.to_yolo_lines(image, CLASS_MAP)
    assert lines == ["3 0.500000 0.500000 0.500000 0.500000"]
    assert dropped == {"unmapped:OC": 1}


def test_boxes_are_clipped_and_degenerate_boxes_dropped(tmp_path):
    image = convert.parse_voc(write_voc(tmp_path / "a.xml", [("LC", -10, 0, 100, 150), ("TC", 5, 5, 5.5, 50)]))
    lines, dropped = convert.to_yolo_lines(image, CLASS_MAP)
    assert lines == ["0 0.250000 0.500000 0.500000 1.000000"]
    assert dropped == {"clipped_kept": 1, "degenerate": 1}


@pytest.mark.parametrize(
    ("objects", "tag"),
    [([("AC", 1, 1, 9, 9)], convert.DAMAGED), ([], convert.CLEAN), ([("RP", 1, 1, 9, 9)], convert.HARD_NEGATIVE)],
)
def test_images_are_tagged(tmp_path, objects, tag):
    assert convert.tag_image(convert.parse_voc(write_voc(tmp_path / "a.xml", objects)), CLASS_MAP) == tag


def test_convert_dir_writes_images_labels_and_records(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "t1.jpg").write_bytes(b"jpeg-bytes")
    xml = write_voc(src / "t1.xml", [])
    records = convert.convert_voc_dir([(src / "t1.jpg", xml)], tmp_path / "out", CLASS_MAP, lambda p: f"x_{p.stem}")
    assert (tmp_path / "out" / "images" / "x_t1.jpg").read_bytes() == b"jpeg-bytes"
    assert (tmp_path / "out" / "labels" / "x_t1.txt").read_text() == ""
    assert records[0]["tag"] == convert.CLEAN
