from pydantic import BaseModel


class QA(BaseModel):
    query: str
    collection: str
