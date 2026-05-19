from fastapi import APIRouter
from app.schemas.forecast import ForecastRequest
from app.services.forecast import process_forecasting

router = APIRouter(prefix="/predict", tags=["Forecasting Engine"])

@router.post("/forecast")
def predict_future_expense(req: ForecastRequest):
    return process_forecasting(req)