from __future__ import annotations

import json

from langgraph import graph
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
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
     human_review_node, 
     revision_router
)
from nodes.scorer import scorer_node
from state import JobMatchingState
import state


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
    graph.add_node("human_review", human_review_node)
    graph.add_node("revision_router", revision_router)
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
    graph.add_edge("end", "human_review")
    graph.add_conditional_edges(
    "human_review",
    revision_router,
    {
        "process_high": "process_high",
        "final": END,
    },
)

    checkpointer = MemorySaver()
    return graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["human_review"]
)




def main() -> None:
    app = build_graph()

    initial_state: JobMatchingState = {
        "candidate_profile": {},
        "list_of_jobs": [],
        "list_of_analyzed_jobs": [],
    }

    config = {"configurable": {"thread_id": "job-session-1"}}

    # 🔹 FIRST RUN (will pause at human_review)
    paused_state = app.invoke(initial_state, config=config)

    print("\n--- PAUSED FOR HUMAN REVIEW ---\n")

    # 🔹 Get paused state
    state = app.get_state(config)
    print("\n--- PAUSED NODE1 ---")
    print(state.next)

    jobs = state.values["list_of_analyzed_jobs"]
    print("\nfirst human review")
    # 🔥 Simulate human decision (put ONE job in revision)
    updated_jobs = []
    for i, job in enumerate(jobs):
        if i == 0:
            decision = "REVISION"   # 👈 one job revision
        else:
            decision = "APPROVED"

        updated_jobs.append(
            job.model_copy(update={"decision": decision})
        )

    # 🔹 Persist review updates to checkpointed graph state
    app.update_state(
        config,
        {"list_of_analyzed_jobs": updated_jobs},
        as_node="human_review",
    )

    # 🔹 RESUME GRAPH
    final_state1 = app.invoke(None, config=config)

    

     # 🔹 Get paused state
    state = app.get_state(config)
    print("\n--- PAUSED NODE2 ---")
    print(state.next)

    # 🔹 SECOND HUMAN REVIEW (after loop pause)
    print("\n--- SECOND HUMAN REVIEW ---\n")

    state = app.get_state(config)

    jobs = state.values["list_of_analyzed_jobs"]

    # 🔥 Now mark ALL as APPROVED (finish flow)
    updated_jobs = [
        job.model_copy(update={"decision": "APPROVED"})
        for job in jobs
    ]

    # 🔹 Update state again
    app.update_state(
        config,
        {"list_of_analyzed_jobs": updated_jobs},
        as_node="human_review",
    )

    # 🔹 FINAL RESUME (third invoke)
    final_state = app.invoke(None, config=config)
    # 🔹 FINAL OUTPUT
    result = {
        "candidate_profile": final_state["candidate_profile"].model_dump(),
        "list_of_jobs": [job.model_dump() for job in final_state["list_of_jobs"]],
        "list_of_analyzed_jobs": [
            analyzed_job.model_dump()
            for analyzed_job in final_state["list_of_analyzed_jobs"]
        ],
    }
    # 🔹 Check final state
    state = app.get_state(config)
    print("\n--- FINAL NODE ---")
    print(state.next)

    print("\n--- FINAL RESULT AFTER HUMAN REVIEW ---\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()