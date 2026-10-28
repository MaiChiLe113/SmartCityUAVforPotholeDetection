"""FastAPI app factory. Start with `python -m run.serve`."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.db import init_db
from app.routers import detections, flight_plans, missions, monitoring, reports
from config import API, PATHS


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # TODO(SCRUM-29, Hai): app.state.detector = build_detector(); app.state.detector.load()
    yield


def create_app() -> FastAPI:
    PATHS.media_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="Pothole UAV API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(API.cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount("/media", StaticFiles(directory=PATHS.media_dir), name="media")

    for module in (reports, detections, flight_plans, missions, monitoring):
        app.include_router(module.router)

    @app.get("/health", tags=["health"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
