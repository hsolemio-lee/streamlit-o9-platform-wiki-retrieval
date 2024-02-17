from pydantic import BaseModel


class EmbeddingInfo(BaseModel):
    collection: str
    file_url: str
    dir_url: str
