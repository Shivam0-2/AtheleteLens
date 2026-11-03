"""Version 1 API routes."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health", tags=["health"])
async def versioned_health_check() -> dict[str, str]:
    """Report that the version 1 API router is responding."""
    return {"status": "healthy", "api_version": "v1"}
