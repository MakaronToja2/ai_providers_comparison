from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from app.config.settings import get_settings
from app.core.routes import router
from app.core.tool_manager import initialize_tools
from app.dashboard.routes import router as dashboard_router
from app.benchmark.storage import Storage


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup."""
    # Initialize tools
    initialize_tools()

    # Initialize database
    storage = Storage()
    await storage.initialize()
    print("Database initialized successfully")

    yield

    # Cleanup (if needed)


app = FastAPI(
    title="LLM Comparison API",
    description="API for comparing different AI providers for pull request analysis",
    version="1.0.0",
    lifespan=lifespan
)

# Include routers
app.include_router(router, prefix="/api/v1")
app.include_router(dashboard_router)

# Mount static files
static_path = Path(__file__).parent / "app" / "dashboard" / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

@app.get("/")
async def root():
    return {"message": "LLM Comparison API is running. Visit /dashboard for the web UI."}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "llm-comparison-api"}

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )