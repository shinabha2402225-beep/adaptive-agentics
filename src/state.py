from __future__ import annotations
from typing import Annotated, List, Optional
from pydantic import BaseModel, Field


def _append(existing: List[str], new: List[str]) -> List[str]:
    return existing + new


class AgentState(BaseModel):
    topic: str = Field(..., description="Research topic provided by the caller.")
    research_notes: List[str] = Field(default_factory=list)
    draft_article: Optional[str] = Field(default=None)
    messages: Annotated[List[str], _append] = Field(default_factory=list)
