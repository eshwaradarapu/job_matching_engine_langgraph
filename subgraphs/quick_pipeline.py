from __future__ import annotations

import json
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from llm_utils import call_openai_text
from models import AnalyzedJob, CandidateProfile, JobListing


class QuickPipelineState(TypedDict):
    candidate_profile: CandidateProfile
    job_listing: JobListing
    analyzed_job: AnalyzedJob
    extracted_requirements: list[str]
    highlighted_skills: list[str]
    quick_summary: str
    


def extract_requirements_node(state: QuickPipelineState) -> QuickPipelineState:
    job_listing = state["job_listing"]
    extracted_requirements = job_listing.required_skills[:5]
    return {**state, "extracted_requirements": extracted_requirements}


def match_skills_node(state: QuickPipelineState) -> QuickPipelineState:
    candidate_skills = {skill.lower() for skill in state["candidate_profile"].skills}
    highlighted_skills = [
        requirement
        for requirement in state["extracted_requirements"]
        if requirement.lower() in candidate_skills
    ]
    return {**state, "highlighted_skills": highlighted_skills}


def quick_summary_node(state: QuickPipelineState) -> QuickPipelineState:
    job = state["analyzed_job"]
    matched = ", ".join(state["highlighted_skills"]) or "limited direct overlap"
    default_summary = (
        f"Strong potential for {job.title} with relevant skill overlap: {matched}.\n"
        f"Recommend quick-tailor application due to MEDIUM fit (score: {job.score}/10)."
    )

    system_prompt = (
        "You are a career assistant. Write exactly 2 concise lines for a medium-match job. "
        "Line 1: concrete fit signal. Line 2: action-oriented recommendation."
    )
    user_payload = {
        "candidate": state["candidate_profile"].model_dump(),
        "job": state["job_listing"].model_dump(),
        "score": job.score,
        "extracted_requirements": state["extracted_requirements"],
        "highlighted_skills": state["highlighted_skills"],
    }
    summary = call_openai_text(
        system_prompt=system_prompt,
        user_prompt=json.dumps(user_payload, ensure_ascii=True),
        default=default_summary,
    )

    updated_job = job.model_copy(
        update={
            "quick_pipeline_requirements": state["extracted_requirements"],
            "quick_pipeline_highlights": state["highlighted_skills"],
            "quick_pipeline_summary": summary,
        }
    )

    return {
        **state,
        "analyzed_job": updated_job,
        "quick_summary": summary,
    }


def build_quick_pipeline_subgraph():
    graph = StateGraph(QuickPipelineState)
    graph.add_node("extract_requirements", extract_requirements_node)
    graph.add_node("match_skills", match_skills_node)
    graph.add_node("quick_summary", quick_summary_node)

    graph.add_edge(START, "extract_requirements")
    graph.add_edge("extract_requirements", "match_skills")
    graph.add_edge("match_skills", "quick_summary")
    graph.add_edge("quick_summary", END)

    return graph.compile()
