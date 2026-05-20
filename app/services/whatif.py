import numpy as np
from app.core.config import model_registry

class WhatIfService:
    @staticmethod
    def process_prediction(user_profile: dict, simulation: dict) -> dict:
        model = model_registry.whatif_model
        metadata = model_registry.whatif_metadata

        if not model or not metadata:
            raise RuntimeError("Model What-If belum dimuat di memori server.")

        # Kalkulasi Fitur Turunan
        expense_to_income_ratio = user_profile['monthly_expenses'] / user_profile['total_income']
        monthly_cashflow = user_profile['total_income'] - user_profile['monthly_expenses']
        debt_to_income_ratio = user_profile['total_debt'] / user_profile['total_income']
        item_to_income_ratio = simulation['item_price'] / user_profile['total_income']
        
        paylater_monthly_installment = (simulation['item_price'] / simulation['paylater_tenor_months']) * (1 + simulation['paylater_interest_rate'])
        paylater_monthly_burden = paylater_monthly_installment / user_profile['total_income']

        # Pemetaan Komplit 38 Fitur
        raw_features = {
            'age': user_profile['age'],
            'total_income': user_profile['total_income'],
            'monthly_expenses': user_profile['monthly_expenses'],
            'expense_to_income_ratio': expense_to_income_ratio,
            'monthly_cashflow': monthly_cashflow,
            'savings_rate': user_profile['savings_rate'] if user_profile.get('savings_rate') is not None else (monthly_cashflow / user_profile['total_income']),
            'current_savings': user_profile['current_savings'],
            'has_emergency_fund': user_profile['has_emergency_fund'],
            'emergency_fund_months': user_profile['emergency_fund_months'],
            'has_kpr': user_profile['has_kpr'],
            'has_vehicle_credit': user_profile['has_vehicle_credit'],
            'pinjol_active': user_profile['pinjol_active'],
            'total_debt': user_profile['total_debt'],
            'debt_to_income_ratio': debt_to_income_ratio,
            'credit_card_utilization': user_profile['credit_card_utilization'],
            'financial_literacy_score': user_profile['financial_literacy_score'],
            
            'item_price': simulation['item_price'],
            'available_cash': simulation['available_cash'],
            'paylater_interest_rate': simulation['paylater_interest_rate'],
            'paylater_tenor_months': simulation['paylater_tenor_months'],
            'paylater_monthly_burden': paylater_monthly_burden,
            'item_to_income_ratio': item_to_income_ratio,

            'employment_type_civil_servant': 1 if user_profile['employment_type'] == 'civil_servant' else 0,
            'employment_type_entrepreneur': 1 if user_profile['employment_type'] == 'entrepreneur' else 0,
            'employment_type_freelance': 1 if user_profile['employment_type'] == 'freelance' else 0,
            'employment_type_gig': 1 if user_profile['employment_type'] == 'gig' else 0,
            'employment_type_not_working': 1 if user_profile['employment_type'] == 'not_working' else 0,
            'employment_type_permanent': 1 if user_profile['employment_type'] == 'permanent' else 0,
            
            'city_tier_tier_1': 1 if user_profile['city_tier'] == 'tier_1' else 0,
            'city_tier_tier_2': 1 if user_profile['city_tier'] == 'tier_2' else 0,
            'city_tier_tier_3': 1 if user_profile['city_tier'] == 'tier_3' else 0,
            
            'paylater_usage_history_frequent': 1 if user_profile['paylater_usage_history'] == 'frequent' else 0,
            'paylater_usage_history_never': 1 if user_profile['paylater_usage_history'] == 'never' else 0,
            'paylater_usage_history_occasional': 1 if user_profile['paylater_usage_history'] == 'occasional' else 0,
            'paylater_usage_history_problematic': 1 if user_profile['paylater_usage_history'] == 'problematic' else 0,
            
            'impulse_spending_tendency_high': 1 if user_profile['impulse_spending_tendency'] == 'high' else 0,
            'impulse_spending_tendency_low': 1 if user_profile['impulse_spending_tendency'] == 'low' else 0,
            'impulse_spending_tendency_medium': 1 if user_profile['impulse_spending_tendency'] == 'medium' else 0
        }

        # Z-Score Standardization
        feature_array = [raw_features.get(col, 0) for col in metadata['feature_columns']]
        scaled_array = [(val - metadata['scaler_mean'][i]) / metadata['scaler_scale'][i] for i, val in enumerate(feature_array)]

        # Keras Model Predict Execution
        input_tensor = np.array([scaled_array], dtype=np.float32)
        prediction = model.predict(input_tensor, verbose=0)
        scores = prediction[0]

        confidence = float(np.max(scores))
        class_index = int(np.argmax(scores))
        recommendation = metadata['class_names'][class_index]

        return {
            "success": True,
            "recommendation": recommendation,
            "confidence": round(confidence, 4),
            "financial_impact": {
                "monthly_cashflow_before": round(monthly_cashflow, 2),
                "monthly_cashflow_after": round(monthly_cashflow - paylater_monthly_installment, 2),
                "paylater_monthly_installment": round(paylater_monthly_installment, 2)
            }
        }