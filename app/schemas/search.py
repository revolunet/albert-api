from typing import List, Literal, Optional, Union
from uuid import UUID
from pydantic import BaseModel, Field, field_validator

from app.schemas.chunks import Chunk
from app.utils.variables import DEFAULT_RAG_K, INTERNET_COLLECTION_DISPLAY_ID, HYBRID_SEARCH_TYPE, LEXICAL_SEARCH_TYPE, SEMANTIC_SEARCH_TYPE


class RagParameters(BaseModel):
    collections: List[Union[UUID, Literal[INTERNET_COLLECTION_DISPLAY_ID]]] = Field(
        description="List of collections ID to search. If not provided, search on all collections.", default=[]
    )
    rff_k: int = Field(default=20, description="k constant in RFF algorithm")
    k: int = Field(gt=0, description="Number of results to return", default=DEFAULT_RAG_K)
    method: Literal[HYBRID_SEARCH_TYPE, LEXICAL_SEARCH_TYPE, SEMANTIC_SEARCH_TYPE] = Field(default=SEMANTIC_SEARCH_TYPE)
    score_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Score of cosine similarity threshold for filtering results")

    @field_validator("collections", mode="before")
    def convert_to_string(cls, collections) -> List[str]:
        return list(set(str(collection) for collection in collections))


class SearchRequest(RagParameters):
    prompt: str = Field(description="Prompt related to the search")

    @field_validator("prompt")
    def blank_string(prompt) -> str:
        if prompt.strip() == "":
            raise ValueError("Prompt cannot be empty")
        return prompt


class Search(BaseModel):
    score: float
    chunk: Chunk
    method: Literal[LEXICAL_SEARCH_TYPE, SEMANTIC_SEARCH_TYPE, f"{LEXICAL_SEARCH_TYPE}/{SEMANTIC_SEARCH_TYPE}"] | None


class Searches(BaseModel):
    object: Literal["list"] = "list"
    data: List[Search]


class Filter(BaseModel):
    pass


class MatchAny(BaseModel):
    any: List[str]


class FieldCondition(BaseModel):
    key: str
    match: MatchAny


class FilterSelector(BaseModel):
    filter: Filter


class PointIdsList(BaseModel):
    points: List[str]
