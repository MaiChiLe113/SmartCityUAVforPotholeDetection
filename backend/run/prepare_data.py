"""Download, clean, split and freeze the dataset (SCRUM-18).  python -m run.prepare_data"""

from ml.build import build_data_yaml, build_raw_dataset, build_splits


def main() -> None:
    raw = build_raw_dataset()
    manifest = build_splits(raw)
    print(f"data.yaml written to {build_data_yaml(manifest)}")


if __name__ == "__main__":
    main()
