from typing import TypedDict

from models import AnalyzedJob, CandidateProfile, JobListing


class JobMatchingState(TypedDict):
    # User requested compact schema: exactly these three core fields.
    candidate_profile: CandidateProfile
    list_of_jobs: list[JobListing]
    list_of_analyzed_jobs: list[AnalyzedJob]
