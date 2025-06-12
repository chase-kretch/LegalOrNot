from pydantic import BaseModel

class AskRequest(BaseModel):
    pdf_id: str
    question: str
    top_k: int = 3

