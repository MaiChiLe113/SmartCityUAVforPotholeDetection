"""Write the grouped prototype train/val lists (SCRUM-38).  python -m run.prototype_split"""

import json

from config import DATA
from config import PROTOTYPE_SPLIT as CFG
from ml.grouping import build_grouped_split


def main() -> None:
    summary = build_grouped_split(CFG.source_dir, CFG.output_dir, CFG.val_share, CFG.seed, DATA.classes)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
