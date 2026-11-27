"""Build new-data batches from an RDD2022 country zip.

python -m run.prepare_batches --zip data/raw/rdd2022_czech/Czech.zip --country Czech --batches 5

Writes data/new/batch_001..005. RDD2022 country zips are inside the FigShare archive
(RDD2022/<Country>.zip, figshare.com/articles/dataset/21431547).
Add one with `dvc add data/new/batch_001` to trigger retraining.
"""

import argparse
import json
from pathlib import Path

from ml.batches import prepare_country_batches


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--zip", type=Path, required=True, help="RDD2022 country zip (kept read-only)")
    p.add_argument("--country", required=True, help="RDD2022 folder/file name, e.g. Czech")
    p.add_argument("--batches", type=int, default=3)
    p.add_argument("--first-index", type=int, default=1, help="number of the first batch folder")
    a = p.parse_args()
    print(json.dumps(prepare_country_batches(a.zip, a.country, a.batches, a.first_index), indent=2))


if __name__ == "__main__":
    main()
