from __future__ import annotations
import pytest
from src.agents import researcher_node, writer_node
from src.graph import build_graph, graph
from src.state import AgentState


class TestAgentState:
    def test_requires_topic(self):
        with pytest.raises(Exception):
            AgentState()

    def test_defaults(self):
        state = AgentState(topic="AI")
        assert state.research_notes == []
        assert state.draft_article is None
        assert state.messages == []


class TestResearcherNode:
    def test_returns_notes(self):
        state = AgentState(topic="quantum computing")
        result = researcher_node(state)
        assert "research_notes" in result
        assert len(result["research_notes"]) > 0

    def test_notes_mention_topic(self):
        topic = "renewable energy"
        state = AgentState(topic=topic)
        result = researcher_node(state)
        combined = " ".join(result["research_notes"]).lower()
        assert topic.lower() in combined

    def test_appends_activity_message(self):
        state = AgentState(topic="robotics")
        result = researcher_node(state)
        assert "Researcher" in result["messages"][0]


class TestWriterNode:
    def _state_with_notes(self, topic: str = "AI safety") -> AgentState:
        researcher_result = researcher_node(AgentState(topic=topic))
        return AgentState(
            topic=topic,
            research_notes=researcher_result["research_notes"],
            messages=researcher_result["messages"],
        )

    def test_returns_draft_article(self):
        state = self._state_with_notes()
        result = writer_node(state)
        assert "draft_article" in result
        assert len(result["draft_article"]) > 50

    def test_article_contains_topic(self):
        topic = "machine learning"
        state = self._state_with_notes(topic)
        result = writer_node(state)
        assert topic.lower() in result["draft_article"].lower()


class TestGraph:
    def test_graph_compiles(self):
        g = build_graph()
        assert g is not None

    def test_full_pipeline(self):
        topic = "LangGraph multi-agent systems"
        result = graph.invoke(AgentState(topic=topic))
        assert result["topic"] == topic
        assert len(result["research_notes"]) > 0
        assert result["draft_article"] is not None

    def test_pipeline_preserves_topic(self):
        topic = "autonomous vehicles"
        result = graph.invoke(AgentState(topic=topic))
        assert result["topic"] == topic
