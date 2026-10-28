from fastapi import HTTPException


def todo(task: str) -> HTTPException:
    """501 response for endpoints not built yet, so the frontend can show a clear message."""
    return HTTPException(status_code=501, detail=f"Not implemented yet: {task}")
