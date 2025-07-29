from fastapi import APIRouter

# Import all route modules
from ..routes.generate import router as generate_router
from ..routes.health import router as health_router
from ..routes.debug import router as debug_router
from ..routes.swe_bench import router as swe_bench_router

# Create main router
router = APIRouter()

# Include all route modules
router.include_router(generate_router, tags=["Generate"])
router.include_router(health_router, tags=["Health"])
router.include_router(debug_router, tags=["Debug"])
router.include_router(swe_bench_router, tags=["SWE-bench"])