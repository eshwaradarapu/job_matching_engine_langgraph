from __future__ import annotations

import json
from typing import TypedDict, Optional

from langgraph.graph import END, START, StateGraph
from models import TailoredResume
from llm_utils import call_openai_text, call_openai_json
from models import AnalyzedJob, CandidateProfile, JobListing


class HighPipelineState(TypedDict):
    candidate_profile: CandidateProfile
    job_listing: JobListing
    analyzed_job: AnalyzedJob
    tailored_resume: Optional[TailoredResume]
    cover_letter: Optional[str]
    quality_score: int
    retry_count: int
    feedback: Optional[str]


# -------------------- RESUME NODE --------------------

def tailor_resume_node(state: HighPipelineState) -> HighPipelineState:
    profile = state["candidate_profile"]
    job = state["job_listing"]
    feedback = state.get("feedback", "")

    system_prompt = (
        "You are a resume expert.\n"
        "Improve the resume based on job description.\n"
        "If feedback is provided, FIX the issues mentioned.\n\n"
        f"Feedback: {feedback}\n\n"
        "Return ONLY JSON:\n"
        "{\n"
        '  "relevant_skills": [list of skills],\n'
        '  "summary": "text"\n'
        "}"
    )

    user_payload = {
        "candidate": {
            "skills": profile.skills,
            "experience_years": profile.experience_years,
            "resume_details": profile.resume_details,
            "title": profile.title,
        },
        "job": {
            "title": job.title,
            "required_skills": job.required_skills,
            "preferred_skills": job.preferred_skills,
            "description": job.description,
        },
    }

    response = call_openai_json(
        system_prompt=system_prompt,
        user_prompt=json.dumps(user_payload, ensure_ascii=True),
        default={
            "relevant_skills": profile.skills[:5],
            "summary": f"{profile.title} with {profile.experience_years} years of experience.",
        },
    )

    tailored_resume = TailoredResume(
        relevant_skills=response.get("relevant_skills", profile.skills[:5]),
        summary=response.get("summary", ""),
    )

    return {**state, "tailored_resume": tailored_resume}


# -------------------- COVER LETTER NODE --------------------

def generate_cover_letter_node(state: HighPipelineState) -> HighPipelineState:
    profile = state["candidate_profile"]
    job = state["job_listing"]
    resume = state.get("tailored_resume")
    feedback = state.get("feedback", "")

    if not resume:
        return {**state, "cover_letter": ""}

    system_prompt = (
        "You are a professional career assistant.\n"
        "Write a strong, concise, and personalized cover letter.\n"
        "If feedback is provided, improve the cover letter accordingly.\n\n"
        f"Feedback: {feedback}\n\n"
        "Rules:\n"
        "- Keep it under 200 words\n"
        "- Make it specific to the job\n"
        "- Highlight relevant skills\n"
        "- Sound natural and confident\n"
        "- Do NOT include placeholders\n"
    )

    user_payload = {
        "candidate": {
            "name": profile.name,
            "title": profile.title,
            "experience_years": profile.experience_years,
        },
        "job": {
            "title": job.title,
            "company": job.company,
            "description": job.description,
        },
        "resume": {
            "skills": resume.relevant_skills,
            "summary": resume.summary,
        },
    }

    cover_letter = call_openai_text(
        system_prompt=system_prompt,
        user_prompt=json.dumps(user_payload, ensure_ascii=True),
        default=(
            f"Dear Hiring Manager,\n\n"
            f"I am excited to apply for the {job.title} role at {job.company}. "
            f"With my experience as a {profile.title} and skills in "
            f"{', '.join(resume.relevant_skills)}, I believe I am a strong fit.\n\n"
            f"Thank you for your consideration.\n"
            f"{profile.name}"
        ),
    )

    return {**state, "cover_letter": cover_letter}


# -------------------- QUALITY NODE --------------------

def quality_check_node(state: HighPipelineState) -> HighPipelineState:
    resume = state.get("tailored_resume")
    cover_letter = state.get("cover_letter")
    job = state["job_listing"]

    if not resume:
        return {**state, "quality_score": 5, "feedback": "Resume missing"}

    system_prompt = (
        "You are a strict recruiter.\n"
        "Evaluate the quality of a job application.\n\n"
        "Return ONLY JSON:\n"
        "{ 'score': number, 'feedback': 'short suggestion' }"
    )

    user_payload = {
        "job": {
            "title": job.title,
            "required_skills": job.required_skills,
        },
        "resume": {
            "skills": resume.relevant_skills,
            "summary": resume.summary,
        },
        "cover_letter": cover_letter,
    }

    response = call_openai_json(
        system_prompt=system_prompt,
        user_prompt=json.dumps(user_payload, ensure_ascii=True),
        default={"score": 6, "feedback": "Improve alignment"},
    )

    return {
        **state,
        "quality_score": response.get("score", 6),
        "retry_count": state.get("retry_count", 0) + 1,
        "feedback": response.get("feedback", ""),
    }


# -------------------- GATE --------------------

def quality_gate(state: HighPipelineState):
    if state.get("retry_count", 0) >= 2:
        return "end"
    if state["quality_score"] < 7:
        return "tailor_resume"
    return "end"


# -------------------- FINAL UPDATE --------------------

def update_job_node(state: HighPipelineState) -> HighPipelineState:
    job = state["analyzed_job"]
    resume = state.get("tailored_resume")

    updated_job = job.model_copy(
        update={
            "tailored_resume": {
                "skills": resume.relevant_skills if resume else [],
                "summary": resume.summary if resume else "",
            },
            "cover_letter": state.get("cover_letter"),
            "quality_score": state.get("quality_score"),
        }
    )

    return {**state, "analyzed_job": updated_job}


# -------------------- GRAPH --------------------

def build_high_pipeline_subgraph():
    graph = StateGraph(HighPipelineState)

    graph.add_node("tailor_resume", tailor_resume_node)
    graph.add_node("generate_cover", generate_cover_letter_node)
    graph.add_node("quality_check", quality_check_node)
    graph.add_node("update_job", update_job_node)

    graph.add_edge(START, "tailor_resume")
    graph.add_edge("tailor_resume", "generate_cover")
    graph.add_edge("generate_cover", "quality_check")

    graph.add_conditional_edges(
        "quality_check",
        quality_gate,
        {
            "tailor_resume": "tailor_resume",
            "end": "update_job",
        },
    )

    graph.add_edge("update_job", END)

    return graph.compile()