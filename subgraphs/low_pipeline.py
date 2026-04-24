from __future__ import annotations

import json
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from llm_utils import call_openai_text
from models import AnalyzedJob


class LowPipelineState(TypedDict):
    analyzed_job: AnalyzedJob
    gap_summary: str
    skip_log: str


def analyze_gap_node(state: LowPipelineState) -> LowPipelineState:
    analyzed_job = state["analyzed_job"]
    missing = ", ".join(analyzed_job.missing_skills) or "general role mismatch"
    default_gap_summary = f"Score {analyzed_job.score}/10 with key gaps: {missing}."

    system_prompt = (
        "You are a recruiting assistant. Summarize why this low-match job should be skipped "
        "in one concise sentence mentioning main gaps."
    )
    user_payload = {
        "job_title": analyzed_job.title,
        "company": analyzed_job.company,
        "score": analyzed_job.score,
        "matched_skills": analyzed_job.matched_skills,
        "missing_skills": analyzed_job.missing_skills,
        "existing_match_details": analyzed_job.match_details,
    }
    gap_summary = call_openai_text(
        system_prompt=system_prompt,
        user_prompt=json.dumps(user_payload, ensure_ascii=True),
        default=default_gap_summary,
    )

    return {**state, "gap_summary": gap_summary}


def skip_log_node(state: LowPipelineState) -> LowPipelineState:
    analyzed_job = state["analyzed_job"]
    skip_log = (
        f"Skipped {analyzed_job.title} at {analyzed_job.company}. "
        f"Reason: {state['gap_summary']}"
    )

    updated_job = analyzed_job.model_copy(
        update={
            "skip_log": skip_log,
        }
    )

    return {
        **state,
        "analyzed_job": updated_job,
        "skip_log": skip_log,
    }


def build_low_pipeline_subgraph():
    graph = StateGraph(LowPipelineState)
    graph.add_node("analyze_gap", analyze_gap_node)
    graph.add_node("skip_log", skip_log_node)

    graph.add_edge(START, "analyze_gap")
    graph.add_edge("analyze_gap", "skip_log")
    graph.add_edge("skip_log", END)

    return graph.compile()
