from fastapi import APIRouter, HTTPException
from app.schemas.whatif import WhatIfRequest, WhatIfResponse
from app.services.whatif_service import WhatIfService

router = APIRouter(prefix="/predict", tags=["What-If Lab Simulation"])

@router.post("/whatif", response_model=WhatIfResponse)
def get_whatif_prediction(payload: WhatIfRequest):
    try:
        result = WhatIfService.process_prediction(
            user_profile=payload.user_profile.model_dump(),
            simulation=payload.simulation.model_dump()
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))