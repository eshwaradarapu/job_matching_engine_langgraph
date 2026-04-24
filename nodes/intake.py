from __future__ import annotations

import json
from pathlib import Path

from models import CandidateProfile, JobListing
from state import JobMatchingState


ROOT_DIR = Path(__file__).resolve().parent.parent


def intake_node(_: JobMatchingState) -> JobMatchingState:
    candidate_data = json.loads((ROOT_DIR / "data" / "candidate.json").read_text(encoding="utf-8"))
    jobs_data = json.loads((ROOT_DIR / "data" / "jobs.json").read_text(encoding="utf-8"))

    candidate_profile = CandidateProfile.model_validate(candidate_data)
    list_of_jobs = [JobListing.model_validate(job) for job in jobs_data]

    return {
        "candidate_profile": candidate_profile,
        "list_of_jobs": list_of_jobs,
        "list_of_analyzed_jobs": [],
    }
