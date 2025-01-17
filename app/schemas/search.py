from typing import List, Literal, Optional, Union
from uuid import UUID
from pydantic import BaseModel, Field, field_validator

from app.schemas.chunks import Chunk
from app.utils.variables import INTERNET_COLLECTION_DISPLAY_ID, HYBRID_SEARCH_TYPE, LEXICAL_SEARCH_TYPE, SEMANTIC_SEARCH_TYPE


class SearchRequest(BaseModel):
    prompt: str
    collections: List[Union[UUID, Literal[INTERNET_COLLECTION_DISPLAY_ID]]]
    rff_k: int = Field(default=20, description="k constant in RFF algorithm")
    k: int = Field(gt=0, description="Number of results to return")
    method: Literal[HYBRID_SEARCH_TYPE, LEXICAL_SEARCH_TYPE, SEMANTIC_SEARCH_TYPE] = Field(default=SEMANTIC_SEARCH_TYPE)
    score_threshold: Optional[float] = Field(0.0, ge=0.0, le=1.0, description="Score of cosine similarity threshold for filtering results")

    @field_validator("prompt")
    def blank_string(prompt):
        if prompt.strip() == "":
            raise ValueError("Prompt cannot be empty")
        return prompt

    @field_validator("collections")
    def convert_to_string(cls, collections):
        if collections is None:
            return []
        return list(set(str(collection) for collection in collections))


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
