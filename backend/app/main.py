"""
AuroraML — Main Application Entry Point
Production-grade Automated Machine Learning Platform.
"""

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.database import init_db

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "AuroraML is an end-to-end automated machine learning platform that enables "
        "businesses to build, deploy, and monitor AI models without data science expertise."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Startup / Shutdown Events ───────────────────────────────────────────────


@app.on_event("startup")
async def startup_event():
    """Initialize database tables and storage directories on startup."""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")

    # Create database tables
    try:
        init_db()
        logger.info("Database tables initialized")
    except Exception as e:
        logger.warning(f"Database init skipped (connect when available): {e}")

    # Create local storage directories
    storage_dirs = [
        os.path.join(settings.LOCAL_STORAGE_PATH, "datasets"),
        os.path.join(settings.LOCAL_STORAGE_PATH, "models"),
    ]
    for d in storage_dirs:
        os.makedirs(d, exist_ok=True)
    logger.info("Storage directories initialized")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down AuroraML")


# ─── Register Routers ────────────────────────────────────────────────────────

from app.api.endpoints import auth, projects, datasets, jobs, models, predictions, monitoring, dashboard, notifications

API_V1_PREFIX = "/api/v1"

app.include_router(auth.router, prefix=API_V1_PREFIX)
app.include_router(dashboard.router, prefix=API_V1_PREFIX)
app.include_router(projects.router, prefix=API_V1_PREFIX)
app.include_router(datasets.router, prefix=API_V1_PREFIX)
app.include_router(jobs.router, prefix=API_V1_PREFIX)
app.include_router(models.router, prefix=API_V1_PREFIX)
app.include_router(predictions.router, prefix=API_V1_PREFIX)
app.include_router(monitoring.router, prefix=API_V1_PREFIX)
app.include_router(notifications.router, prefix=API_V1_PREFIX)


@app.get("/api/v1/health")
def health_check():
    return {"status": "healthy", "version": settings.APP_VERSION}


# ─── Root & Health Endpoints ─────────────────────────────────────────────────


@app.get("/", tags=["Root"])
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "version": settings.APP_VERSION}


# ─── Global Exception Handler ────────────────────────────────────────────────


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
