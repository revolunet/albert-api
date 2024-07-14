from typing import Literal, List

from pydantic import BaseModel


class Collection(BaseModel):
    object: Literal["collection"]
    name: str
    type: Literal["public", "user"]


class CollectionResponse(BaseModel):
    object: Literal["list"]
    data: List[Collection]
