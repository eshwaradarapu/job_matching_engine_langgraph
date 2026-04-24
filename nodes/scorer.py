from __future__ import annotations

import json
import re

from llm_utils import call_openai_json
from models import AnalyzedJob
from state import JobMatchingState


def _normalize(values: list[str]) -> set[str]:
    return {value.strip().lower() for value in values}


RELATED_SKILLS: dict[str, set[str]] = {
    "langgraph": {"langchain", "crewai"},
    "llm apis": {"langchain", "crewai"},
    "pydantic": {"fastapi"},
    "k8s": {"docker"},
}


def _tokenize(text: str) -> set[str]:
    return {token for token in re.split(r"[^a-zA-Z0-9]+", text.lower()) if token}


def _score_to_category(score: int) -> str:
    if score >= 8:
        return "HIGH"
    if score >= 5:
        return "MEDIUM"
    return "LOW"


def _score_job_heuristic(state: JobMatchingState, job_index: int) -> AnalyzedJob:
    candidate = state["candidate_profile"]
    job = state["list_of_jobs"][job_index]

    candidate_skills = _normalize(candidate.skills)
    required_skills = _normalize(job.required_skills)

    exact_matches = required_skills.intersection(candidate_skills)
    related_matches = set()
    missing_required = []

    for required_skill in required_skills:
        if required_skill in exact_matches:
            continue

        related_candidates = RELATED_SKILLS.get(required_skill, set())
        if related_candidates.intersection(candidate_skills):
            related_matches.add(required_skill)
        else:
            missing_required.append(required_skill)

    matched_required = sorted(exact_matches.union(related_matches))
    missing_required.sort()

    # 6 points: required skill overlap
    weighted_matched_count = len(exact_matches) + (0.5 * len(related_matches))
    overlap_ratio = weighted_matched_count / max(1, len(required_skills))
    overlap_points = overlap_ratio * 6

    # 2 points: experience alignment
    if candidate.experience_years >= job.min_experience_years:
        experience_points = 2
    else:
        gap = job.min_experience_years - candidate.experience_years
        experience_points = max(0, 2 - gap)

    # 2 points: role/title alignment (simple keyword overlap)
    preference_tokens = _tokenize(f"{candidate.preferences.role_type} {candidate.title}")
    job_tokens = _tokenize(f"{job.title} {job.description}")
    role_points = 1.5 if preference_tokens.intersection(job_tokens) else 0.5

    raw_score = overlap_points + experience_points + role_points
    score = max(1, min(10, int(round(raw_score))))

    category = _score_to_category(score)

    details = (
        f"Matched required skills: {len(matched_required)}/{len(required_skills)}; "
        f"experience fit: {candidate.experience_years} vs min {job.min_experience_years}; "
        f"computed score: {score}."
    )

    return AnalyzedJob(
        job_id=job.id,
        title=job.title,
        company=job.company,
        location=job.location,
        score=score,
        category=category,
        matched_skills=matched_required,
        missing_skills=missing_required,
        match_details=details,
    )


def _score_job_with_llm(state: JobMatchingState, job_index: int) -> AnalyzedJob:
    candidate = state["candidate_profile"]
    job = state["list_of_jobs"][job_index]

    fallback = _score_job_heuristic(state, job_index)

    system_prompt = (
        "You are a recruiting scoring assistant. "
        "Return strict JSON only with keys: score, category, matched_skills, "
        "missing_skills, match_details. "
        "Scoring must be 1-10 integer. Category must be HIGH, MEDIUM, or LOW."
    )

    user_payload = {
        "candidate_profile": candidate.model_dump(),
        "job_listing": job.model_dump(),
        "rubric": {
            "skill_overlap_weight": 0.6,
            "experience_weight": 0.2,
            "role_alignment_weight": 0.2,
            "category_bands": {
                "HIGH": "8-10",
                "MEDIUM": "5-7",
                "LOW": "1-4",
            },
        },
    }

    llm_result = call_openai_json(
        system_prompt=system_prompt,
        user_prompt=json.dumps(user_payload, ensure_ascii=True),
        default=fallback.model_dump(),
    )

    raw_score = llm_result.get("score", fallback.score)
    try:
        score = int(raw_score)
    except (TypeError, ValueError):
        score = fallback.score
    score = max(1, min(10, score))

    raw_category = str(llm_result.get("category", "")).upper()
    if raw_category not in {"HIGH", "MEDIUM", "LOW"}:
        raw_category = _score_to_category(score)

    matched_skills_raw = llm_result.get("matched_skills", fallback.matched_skills)
    missing_skills_raw = llm_result.get("missing_skills", fallback.missing_skills)

    matched_skills = [str(skill).strip().lower() for skill in matched_skills_raw if str(skill).strip()]
    missing_skills = [str(skill).strip().lower() for skill in missing_skills_raw if str(skill).strip()]

    if not matched_skills:
        matched_skills = fallback.matched_skills
    if not missing_skills:
        missing_skills = fallback.missing_skills

    match_details = str(llm_result.get("match_details") or fallback.match_details)

    return AnalyzedJob(
        job_id=job.id,
        title=job.title,
        company=job.company,
        location=job.location,
        score=score,
        category=raw_category,
        matched_skills=sorted(set(matched_skills)),
        missing_skills=sorted(set(missing_skills)),
        match_details=match_details,
    )


def scorer_node(state: JobMatchingState) -> JobMatchingState:
    analyzed_jobs = [_score_job_with_llm(state, index) for index in range(len(state["list_of_jobs"]))]
    analyzed_jobs.sort(key=lambda item: item.score, reverse=True)

    return {
        **state,
        "list_of_analyzed_jobs": analyzed_jobs,
    }
