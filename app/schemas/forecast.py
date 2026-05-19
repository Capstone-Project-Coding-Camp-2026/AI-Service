from pydantic import BaseModel

class ForecastRequest(BaseModel):
    lag1_total_expense: float
    lag2_total_expense: float
    lag3_total_expense: float
    roll3_mean_expense: float
    roll6_mean_expense: float
    roll3_std_expense: float
    lag1_monthly_income: float
    lag1_savings_rate: float
    lag1_expense_growth: float
    bulan_sin: float
    bulan_cos: float
    persona_id: float