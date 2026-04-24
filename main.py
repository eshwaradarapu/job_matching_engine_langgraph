from __future__ import annotations

import json

from langgraph.graph import END, START, StateGraph

from nodes.intake import intake_node
from nodes.router import (
    end_node,
    process_high_node,
    process_low_node,
    process_medium_node,
    route_high,
    route_low,
    route_low_gate_node,
    route_medium,
    route_medium_gate_node,
    start_routing_node,
)
from nodes.scorer import scorer_node
from state import JobMatchingState


def build_graph():
    graph = StateGraph(JobMatchingState)

    graph.add_node("intake", intake_node)
    graph.add_node("scorer", scorer_node)

    # Stage 2 routing gates + processing nodes.
    graph.add_node("route_start", start_routing_node)
    graph.add_node("process_high", process_high_node)
    graph.add_node("route_medium", route_medium_gate_node)
    graph.add_node("process_medium", process_medium_node)
    graph.add_node("route_low", route_low_gate_node)
    graph.add_node("process_low", process_low_node)
    graph.add_node("end", end_node)

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "scorer")
    graph.add_edge("scorer", "route_start")

    graph.add_conditional_edges(
        "route_start",
        route_high,
        {
            "process_high": "process_high",
            "route_medium": "route_medium",
        },
    )

    graph.add_edge("process_high", "route_medium")

    graph.add_conditional_edges(
        "route_medium",
        route_medium,
        {
            "process_medium": "process_medium",
            "route_low": "route_low",
        },
    )

    graph.add_edge("process_medium", "route_low")

    graph.add_conditional_edges(
        "route_low",
        route_low,
        {
            "process_low": "process_low",
            "end": "end",
        },
    )

    graph.add_edge("process_low", "end")
    graph.add_edge("end", END)

    return graph.compile()


def main() -> None:
    app = build_graph()

    initial_state: JobMatchingState = {
        "candidate_profile": {},  # intake_node fills this
        "list_of_jobs": [],       # intake_node fills this
        "list_of_analyzed_jobs": [],
    }

    final_state = app.invoke(initial_state)

    result = {
        "candidate_profile": final_state["candidate_profile"].model_dump(),
        "list_of_jobs": [job.model_dump() for job in final_state["list_of_jobs"]],
        "list_of_analyzed_jobs": [
            analyzed_job.model_dump() for analyzed_job in final_state["list_of_analyzed_jobs"]
        ],
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
