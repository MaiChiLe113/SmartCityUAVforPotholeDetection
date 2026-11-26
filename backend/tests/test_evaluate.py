from ml.evaluate import partition_val_list

GROUND = "/d/images/train/rdd_Japan_Japan_000001.jpg"
RDD_DRONE = "/d/images/val/rdd_China_Drone_China_Drone_000002.jpg"
RDV = "/d/images/val/rdv_285_png_jpg.rf.cc33.jpg"
UAV = "/d/images/train/uav_lr_7_top_left.jpg"


def test_val_list_is_partitioned_by_view_and_source():
    subsets = partition_val_list([GROUND, RDD_DRONE, RDV, UAV])
    assert subsets["all"] == [GROUND, RDD_DRONE, RDV, UAV]
    assert subsets["ground"] == [GROUND]
    assert subsets["aerial"] == [RDD_DRONE, RDV, UAV]
    assert subsets["source_rdd"] == [GROUND, RDD_DRONE]
    assert subsets["source_rdv"] == [RDV]
    assert subsets["source_uav"] == [UAV]


def test_views_cover_every_image_once():
    paths = [GROUND, RDD_DRONE, RDV, UAV]
    subsets = partition_val_list(paths)
    assert sorted(subsets["ground"] + subsets["aerial"]) == sorted(paths)
