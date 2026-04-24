from __future__ import annotations

from typing import Literal

from state import JobMatchingState
from subgraphs.low_pipeline import build_low_pipeline_subgraph
from subgraphs.quick_pipeline import build_quick_pipeline_subgraph


QUICK_PIPELINE_APP = build_quick_pipeline_subgraph()
LOW_PIPELINE_APP = build_low_pipeline_subgraph()


def start_routing_node(state: JobMatchingState) -> JobMatchingState:
    return state


def route_high(state: JobMatchingState) -> Literal["process_high", "route_medium"]:
    has_high = any(job.category == "HIGH" for job in state["list_of_analyzed_jobs"])
    return "process_high" if has_high else "route_medium"


def process_high_node(state: JobMatchingState) -> JobMatchingState:
    updated_jobs = []
    for job in state["list_of_analyzed_jobs"]:
        if job.category == "HIGH":
            updated_jobs.append(
                job.model_copy(
                    update={
                        "stage2_route": "FULL_PIPELINE",
                        "stage2_reason": "HIGH category (8-10): route to detailed/full pipeline.",
                    }
                )
            )
        else:
            updated_jobs.append(job)

    return {**state, "list_of_analyzed_jobs": updated_jobs}


def route_medium_gate_node(state: JobMatchingState) -> JobMatchingState:
    return state


def route_medium(state: JobMatchingState) -> Literal["process_medium", "route_low"]:
    has_medium = any(job.category == "MEDIUM" for job in state["list_of_analyzed_jobs"])
    return "process_medium" if has_medium else "route_low"


def process_medium_node(state: JobMatchingState) -> JobMatchingState:
    updated_jobs = []
    jobs_by_id = {job.id: job for job in state["list_of_jobs"]}

    for job in state["list_of_analyzed_jobs"]:
        if job.category == "MEDIUM":
            job_listing = jobs_by_id.get(job.job_id)
            if job_listing is None:
                updated_jobs.append(
                    job.model_copy(
                        update={
                            "stage2_route": "QUICK_PIPELINE",
                            "stage2_reason": (
                                "MEDIUM category (5-7): quick pipeline selected, "
                                "but no matching job listing was found."
                            ),
                        }
                    )
                )
                continue

            try:
                subgraph_result = QUICK_PIPELINE_APP.invoke(
                    {
                        "candidate_profile": state["candidate_profile"],
                        "job_listing": job_listing,
                        "analyzed_job": job,
                        "extracted_requirements": [],
                        "highlighted_skills": [],
                        "quick_summary": "",
                    }
                )

                enriched_job = subgraph_result["analyzed_job"].model_copy(
                    update={
                        "stage2_route": "QUICK_PIPELINE",
                        "stage2_reason": (
                            "MEDIUM category (5-7): routed through quick_pipeline "
                            "subgraph (extract -> match -> summary)."
                        ),
                    }
                )
                updated_jobs.append(enriched_job)
            except Exception as exc:
                updated_jobs.append(
                    job.model_copy(
                        update={
                            "stage2_route": "QUICK_PIPELINE",
                            "stage2_reason": (
                                "MEDIUM category (5-7): quick pipeline failed with "
                                f"error: {exc}"
                            ),
                        }
                    )
                )
        else:
            updated_jobs.append(job)

    return {**state, "list_of_analyzed_jobs": updated_jobs}


def route_low_gate_node(state: JobMatchingState) -> JobMatchingState:
    return state


def route_low(state: JobMatchingState) -> Literal["process_low", "end"]:
    has_low = any(job.category == "LOW" for job in state["list_of_analyzed_jobs"])
    return "process_low" if has_low else "end"


def process_low_node(state: JobMatchingState) -> JobMatchingState:
    updated_jobs = []
    for job in state["list_of_analyzed_jobs"]:
        if job.category == "LOW":
            try:
                subgraph_result = LOW_PIPELINE_APP.invoke(
                    {
                        "analyzed_job": job,
                        "gap_summary": "",
                        "skip_log": "",
                    }
                )

                enriched_job = subgraph_result["analyzed_job"].model_copy(
                    update={
                        "stage2_route": "SKIP",
                        "stage2_reason": (
                            "LOW category (1-4): routed through low_pipeline "
                            "subgraph (gap analysis -> skip log)."
                        ),
                    }
                )
                updated_jobs.append(enriched_job)
            except Exception as exc:
                updated_jobs.append(
                    job.model_copy(
                        update={
                            "stage2_route": "SKIP",
                            "stage2_reason": (
                                "LOW category (1-4): skip pipeline failed with "
                                f"error: {exc}"
                            ),
                        }
                    )
                )
        else:
            updated_jobs.append(job)

    return {**state, "list_of_analyzed_jobs": updated_jobs}


def end_node(state: JobMatchingState) -> JobMatchingState:
    return state
