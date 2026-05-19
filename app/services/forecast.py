import numpy as np
from fastapi import HTTPException
from app.core.config import model_registry
from app.schemas.forecast import ForecastRequest

def process_forecasting(req: ForecastRequest):
    if model_registry.forecast_model is None or model_registry.forecast_metadata is None:
        raise HTTPException(status_code=500, detail="Model Forecasting belum siap.")
    
    try:
        metadata = model_registry.forecast_metadata
        mean = np.array(metadata.get("scaler_mean") or metadata.get("mean"), dtype=np.float32)
        scale = np.array(metadata.get("scaler_scale") or metadata.get("std"), dtype=np.float32)
        y_scale = float(metadata.get("y_scale", 1000000.0))
        
        feature_values = [
            req.lag1_total_expense, req.lag2_total_expense, req.lag3_total_expense,
            req.roll3_mean_expense, req.roll6_mean_expense, req.roll3_std_expense,
            req.lag1_monthly_income, req.lag1_savings_rate, req.lag1_expense_growth,
            req.bulan_sin, req.bulan_cos, req.persona_id
        ]
        
        x = np.array([feature_values], dtype=np.float32)
        x_scaled = (x - mean) / scale
        
        pred_scaled = model_registry.forecast_model.predict(x_scaled, verbose=0).ravel()[0]
        predicted_rupiah = float(pred_scaled * y_scale)
        
        return {
            "success": True,
            "predicted_total_expense": round(predicted_rupiah, 2),
            "currency": "IDR",
            "model_metrics": metadata.get("metrics", {})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal prediksi Forecasting: {str(e)}")