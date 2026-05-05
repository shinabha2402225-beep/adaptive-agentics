from __future__ import annotations
from langgraph.graph import END, StateGraph
from .agents import researcher_node, writer_node
from .state import AgentState


def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)
    builder.add_node("researcher", researcher_node)
    builder.add_node("writer", writer_node)
    builder.set_entry_point("researcher")
    builder.add_edge("researcher", "writer")
    builder.add_edge("writer", END)
    return builder.compile()


graph = build_graph()
