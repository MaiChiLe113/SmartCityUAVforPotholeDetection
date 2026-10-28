"""Train all candidates with identical settings and write the comparison (SCRUM-25).
python -m run.compare_models"""

from config import PATHS, TRAIN
from ml.build import build_data_yaml_from_disk, build_model, compile_train_args
from ml.train import compare, train_one


def main() -> None:
    data_yaml = build_data_yaml_from_disk()
    results = [train_one(build_model(m), compile_train_args(m, data_yaml)) for m in TRAIN.candidates]
    print(f"comparison written to {compare(results, PATHS.reports_dir / 'model_comparison.csv')}")


if __name__ == "__main__":
    main()
