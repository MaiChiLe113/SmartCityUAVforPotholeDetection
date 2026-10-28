"""MLflow model registry: register, champion/challenger aliases, rollback. Owner: Chi (SCRUM-28)."""

from __future__ import annotations

from pathlib import Path

from config import MLOPS, MlopsConfig


class ModelRegistry:
    def __init__(self, cfg: MlopsConfig = MLOPS) -> None:
        self.cfg = cfg

    def register(self, weights: Path, run_id: str, tags: dict[str, str]) -> str:
        """Register weights as a new model version linked to its run, data version and git commit.
        Returns the version number."""
        raise NotImplementedError("TODO(SCRUM-28, Chi): register model version")

    def set_alias(self, version: str, alias: str) -> None:
        """Move the "champion" or "challenger" alias to `version`."""
        raise NotImplementedError("TODO(SCRUM-28, Chi): set alias")

    def champion_weights(self) -> Path:
        """Download (or locate) the champion weights for serving."""
        raise NotImplementedError("TODO(SCRUM-28, Chi): fetch champion weights")

    def rollback(self) -> str:
        """Move the champion alias back to the previous champion. Returns that version."""
        raise NotImplementedError("TODO(SCRUM-28, Chi): rollback champion")
