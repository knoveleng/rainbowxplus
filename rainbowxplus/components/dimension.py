from typing import List, Optional, Tuple
from pydantic import BaseModel, Field


class Element(BaseModel):
    name: str
    description: Optional[str] = None


class Dimension(BaseModel):
    name: str
    elements: List[Element] = Field(default_factory=list)
