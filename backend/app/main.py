"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title="AthleteLens API",
    description="Preliminary sports performance assessment API.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=[
        "Accept-Ranges", "Content-Range", "Content-Length",
        "Content-Disposition", "ETag", "X-Assessment-ID",
    ],
)


@app.get("/health", tags=["health"])
async def health_check() -> dict[str, str]:
    """Report that the API process is responding."""
    return {"status": "healthy"}


app.include_router(api_router)
