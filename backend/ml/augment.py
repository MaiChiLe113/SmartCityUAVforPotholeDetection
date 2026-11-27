"""Training-only augmentation (§3.1 Table 3.2). Owner: Hai (SCRUM-19)."""

from __future__ import annotations

from config import AUGMENT, AugmentConfig


def to_ultralytics_hyp(cfg: AugmentConfig = AUGMENT) -> dict:
    """Map AugmentConfig to Ultralytics train() keyword arguments (hsv_v, fliplr, degrees, ...).

    Blur/noise are not native Ultralytics args: add them through Albumentations if needed.
    """
    return {
        "hsv_v": cfg.hsv_v,
        "hsv_s": cfg.hsv_s,
        "fliplr": cfg.fliplr,
        "degrees": cfg.degrees,
        "scale": cfg.scale,
        "mosaic": cfg.mosaic,
    }
