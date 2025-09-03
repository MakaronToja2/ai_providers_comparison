from fastapi import FastAPI
from app.config.settings import get_settings
from app.core.routes import router
from app.core.tool_manager import initialize_tools

app = FastAPI(
    title="LLM Comparison API",
    description="API for comparing different AI providers for pull request analysis",
    version="1.0.0"
)

# Initialize tools on startup
initialize_tools()

app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "LLM Comparison API is running"}

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