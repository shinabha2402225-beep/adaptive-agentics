from __future__ import annotations
import sys
from src import AgentState, graph


def run(topic: str) -> None:
    initial_state = AgentState(topic=topic)
    result = graph.invoke(initial_state)
    print("\n" + "=" * 60)
    print(f"TOPIC: {result['topic']}")
    print("=" * 60)
    print("\nRESEARCH NOTES:")
    for note in result["research_notes"]:
        print(f"  - {note}")
    print("\nDRAFT ARTICLE:")
    print(result["draft_article"])


if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Agentic AI systems"
    run(topic)
