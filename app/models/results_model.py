from pydantic import BaseModel


class SearchResult(BaseModel):
    rank: int
    doc: str
    score: float
