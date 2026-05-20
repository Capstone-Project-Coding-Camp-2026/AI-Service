from pydantic import BaseModel
from typing import Optional

class UserProfileSchema(BaseModel):
    age: int
    total_income: float
    monthly_expenses: float
    current_savings: float
    has_emergency_fund: int
    emergency_fund_months: int
    has_kpr: int
    has_vehicle_credit: int
    pinjol_active: int
    total_debt: float
    credit_card_utilization: float
    financial_literacy_score: float
    employment_type: str
    city_tier: str
    paylater_usage_history: str
    impulse_spending_tendency: str
    savings_rate: Optional[float] = None

class SimulationSchema(BaseModel):
    item_price: float
    available_cash: float
    paylater_interest_rate: float
    paylater_tenor_months: int

class WhatIfRequest(BaseModel):
    user_profile: UserProfileSchema
    simulation: SimulationSchema

class FinancialImpactResponse(BaseModel):
    monthly_cashflow_before: float
    monthly_cashflow_after: float
    paylater_monthly_installment: float

class WhatIfResponse(BaseModel):
    success: bool
    recommendation: str
    confidence: float
    financial_impact: FinancialImpactResponse