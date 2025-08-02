from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.slices.prediction.routers import router as prediction_router
from app.slices.time_statistics.routers import router as time_statistics_router
from app.slices.tracking_statistics.routers import router as tracking_statistics_router
from app.slices.db_statistics.routers import router as db_statistics_router
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prediction_router)
app.include_router(time_statistics_router)
app.include_router(tracking_statistics_router)
app.include_router(db_statistics_router)
