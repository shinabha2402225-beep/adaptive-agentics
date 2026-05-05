from __future__ import annotations
import os
from typing import Any, Dict
from .state import AgentState


def _get_llm():
    try:
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    except ImportError:
        return None


def researcher_node(state: AgentState) -> Dict[str, Any]:
    llm = _get_llm()
    if llm:
        prompt = f"Produce 5 bullet-point facts about: {state.topic}"
        response = llm.invoke(prompt)
        notes = [line.strip() for line in response.content.splitlines() if line.strip()]
    else:
        notes = [
            f"[STUB] Key fact 1 about '{state.topic}'",
            f"[STUB] Key fact 2 about '{state.topic}'",
            f"[STUB] Key fact 3 about '{state.topic}'",
        ]
    return {
        "research_notes": notes,
        "messages": [f"Researcher completed. Produced {len(notes)} notes."],
    }


def writer_node(state: AgentState) -> Dict[str, Any]:
    llm = _get_llm()
    if llm:
        notes_text = "\n".join(state.research_notes)
        prompt = f"Write a short article about '{state.topic}'.\n\nNotes:\n{notes_text}"
        response = llm.invoke(prompt)
        article = response.content.strip()
    else:
        notes_summary = "; ".join(state.research_notes[:3])
        article = (
            f"# {state.topic.title()}\n\n"
            f"[STUB ARTICLE] Research summary: {notes_summary}."
        )
    return {
        "draft_article": article,
        "messages": ["Writer completed. Draft article produced."],
    }
