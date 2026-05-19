from fastapi import APIRouter
from app.schemas.classify import ClassifyRequest
from app.services.classify import process_classification

router = APIRouter(prefix="/predict", tags=["NLP Classification"])

@router.post("/classify")
def predict_category(req: ClassifyRequest):
    # Router hanya meneruskan request ke Service
    return process_classification(req)