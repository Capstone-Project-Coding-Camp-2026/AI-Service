from pydantic import BaseModel

class ClassifyRequest(BaseModel):
    desc: str