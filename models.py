from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CandidatePreferences(BaseModel):
    role_type: str
    location: str
    min_match_score: int = Field(ge=1, le=10)


class CandidateProfile(BaseModel):
    name: str
    title: str
    experience_years: int = Field(ge=0)
    skills: list[str]
    resume_details: str
    preferences: CandidatePreferences


class JobListing(BaseModel):
    id: str
    title: str
    company: str
    location: str
    required_skills: list[str]
    preferred_skills: list[str] = Field(default_factory=list)
    min_experience_years: int = Field(ge=0)
    description: str


class AnalyzedJob(BaseModel):
    job_id: str
    title: str
    company: str
    location: str
    score: int = Field(ge=1, le=10)
    category: Literal["HIGH", "MEDIUM", "LOW"]
    matched_skills: list[str]
    missing_skills: list[str]
    match_details: str
    stage2_route: Literal["FULL_PIPELINE", "QUICK_PIPELINE", "SKIP"] | None = None
    stage2_reason: str | None = None
    quick_pipeline_requirements: list[str] | None = None
    quick_pipeline_highlights: list[str] | None = None
    quick_pipeline_summary: str | None = None
    skip_log: str | None = None
