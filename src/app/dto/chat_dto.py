from pydantic import BaseModel
from typing import List, Dict


class ChatDTO(BaseModel):
    query: str
    chat_history: List[Dict[str, str]]
    collection: str
