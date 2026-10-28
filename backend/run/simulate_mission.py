"""Fly an approved mission in the simulator and upload edge results (SCRUM-33, SCRUM-34).
python -m run.simulate_mission --mission-id 1 [--api http://localhost:8000]"""

import argparse

from config import API


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mission-id", type=int, required=True)
    p.add_argument("--api", default=f"http://localhost:{API.port}")
    p.parse_args()
    # TODO(SCRUM-33, Hoan): GET the mission's approved plan, build SimulatedUAV(route, api, id),
    # feed uav.fly() frames into EdgeRunner(...).run(), then EdgeRunner.upload().
    raise NotImplementedError("TODO(SCRUM-33, Hoan): run a simulated mission")


if __name__ == "__main__":
    main()
