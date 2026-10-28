import os
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

# Tests use a throwaway database so they never touch data/app.db.
os.environ.setdefault("DATABASE_URL", f"sqlite:///{BACKEND / 'data' / 'test.db'}")
