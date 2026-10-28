"""Start the API.  python -m run.serve [--reload]"""

import argparse

import uvicorn

from config import API


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reload", action="store_true")
    a = p.parse_args()
    uvicorn.run("app.main:create_app", factory=True, host=API.host, port=API.port, reload=a.reload)


if __name__ == "__main__":
    main()
