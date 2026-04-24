# Job Matching Engine (LangGraph)

A LangGraph-based Job Matching and Routing Engine that implements:

1. Stage 1: Job intake and scoring
2. Stage 2: Category-based routing

This project uses a compact top-level state schema with exactly three fields:

1. candidate_profile
2. list_of_jobs
3. list_of_analyzed_jobs

The implementation includes OpenAI-backed scoring and text generation with safe fallback behavior when API access is unavailable.

## What This Project Does

Given one candidate profile and multiple jobs, the graph:

1. Loads and validates input data.
2. Scores each job on a 1-10 fit scale.
3. Classifies each job into HIGH, MEDIUM, or LOW.
4. Routes jobs by category:
1. HIGH -> marked for full pipeline.
2. MEDIUM -> processed by a quick subgraph.
3. LOW -> processed by a skip/log subgraph.
5. Returns enriched analyzed jobs with route decisions and generated summaries/logs.

## Architecture

Main graph flow:

1. START -> intake
2. intake -> scorer
3. scorer -> route_start
4. route_start -> process_high or route_medium
5. route_medium -> process_medium or route_low
6. route_low -> process_low or end
7. end -> END

Subgraphs included:

1. MEDIUM quick pipeline subgraph:
1. extract_requirements -> match_skills -> quick_summary
2. LOW skip/log subgraph:
1. analyze_gap -> skip_log

Rendered graph files:

1. graph_full_with_subgraphs.png
2. graph_full_with_subgraphs.svg
3. graph_full_with_subgraphs.mmd

## Schema

### 1) Graph State Schema (Top-Level)

Defined in state.py as a TypedDict:

```python
class JobMatchingState(TypedDict):
    candidate_profile: CandidateProfile
    list_of_jobs: list[JobListing]
    list_of_analyzed_jobs: list[AnalyzedJob]
```

This is the shared state object passed between all LangGraph nodes.

### 2) Pydantic Models

Defined in models.py.

#### CandidatePreferences

```python
class CandidatePreferences(BaseModel):
    role_type: str
    location: str
    min_match_score: int = Field(ge=1, le=10)
```

#### CandidateProfile

```python
class CandidateProfile(BaseModel):
    name: str
    title: str
    experience_years: int = Field(ge=0)
    skills: list[str]
    resume_details: str
    preferences: CandidatePreferences
```

#### JobListing

```python
class JobListing(BaseModel):
    id: str
    title: str
    company: str
    location: str
    required_skills: list[str]
    preferred_skills: list[str] = Field(default_factory=list)
    min_experience_years: int = Field(ge=0)
    description: str
```

#### AnalyzedJob

```python
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
```

### 3) Example Final State Shape

```json
{
  "candidate_profile": {
    "name": "Ravi Kumar",
    "title": "Python Developer",
    "experience_years": 3,
    "skills": ["Python", "FastAPI", "PostgreSQL"],
    "resume_details": "...",
    "preferences": {
      "role_type": "AI/ML Engineer",
      "location": "Remote or Hyderabad",
      "min_match_score": 5
    }
  },
  "list_of_jobs": [
    {
      "id": "job-1",
      "title": "AI Agent Developer",
      "company": "TechCorp",
      "location": "Remote",
      "required_skills": ["Python", "CrewAI"],
      "preferred_skills": ["Playwright"],
      "min_experience_years": 2,
      "description": "..."
    }
  ],
  "list_of_analyzed_jobs": [
    {
      "job_id": "job-1",
      "title": "AI Agent Developer",
      "company": "TechCorp",
      "location": "Remote",
      "score": 8,
      "category": "HIGH",
      "matched_skills": ["python", "crewai"],
      "missing_skills": ["langgraph"],
      "match_details": "...",
      "stage2_route": "FULL_PIPELINE",
      "stage2_reason": "HIGH category...",
      "quick_pipeline_requirements": null,
      "quick_pipeline_highlights": null,
      "quick_pipeline_summary": null,
      "skip_log": null
    }
  ]
}
```

## Stage Details

### Stage 1: Intake and Scoring

1. Intake node reads data files from data/candidate.json and data/jobs.json.
2. Candidate and jobs are validated with Pydantic model_validate.
3. Scorer processes each job and returns:
1. score (1-10)
2. category (HIGH/MEDIUM/LOW)
3. matched_skills / missing_skills
4. match_details
4. Jobs are ranked by score descending.

Scoring uses OpenAI JSON output first and falls back to deterministic heuristic scoring if needed.

### Stage 2: Routing and Category Processing

1. HIGH jobs:
1. Marked as stage2_route = FULL_PIPELINE.
2. MEDIUM jobs:
1. Routed through quick subgraph.
2. Produces extracted requirements, highlighted skills, and 2-line summary.
3. LOW jobs:
1. Routed through low subgraph.
2. Produces gap summary and skip log.

## OpenAI Configuration

This project reads model settings from .env in project root.

Required variables:

```env
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4.1-mini
```

Notes:

1. .env is ignored by git via .gitignore.
2. If OPENAI_API_KEY is missing or calls fail, fallback logic still returns valid outputs.

## Project Structure

```text
langgraph-assignment/
  data/
    candidate.json
    jobs.json
  nodes/
    __init__.py
    intake.py
    scorer.py
    router.py
  subgraphs/
    __init__.py
    quick_pipeline.py
    low_pipeline.py
  .gitignore
  llm_utils.py
  main.py
  models.py
  README.md
  requirements.txt
  state.py
  graph_full_with_subgraphs.mmd
  graph_full_with_subgraphs.png
  graph_full_with_subgraphs.svg
```

## Setup and Run

1. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Create .env in project root:

```env
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4.1-mini
```

3. Run:

```bash
python main.py
```

4. Output:

The program prints final graph state JSON with analyzed job decisions and subgraph enrichments.

## LangGraph Concepts Used

1. StateGraph with typed shared state.
2. Node-based orchestration.
3. Conditional routing via add_conditional_edges.
4. Compiled subgraphs for MEDIUM and LOW processing.
5. Deterministic state updates through immutable-style model_copy.
6. Error-safe fallback behavior for external LLM calls.

## Current Scope

Implemented:

1. Stage 1
2. Stage 2
3. Medium quick subgraph
4. Low skip/log subgraph

Not yet implemented:

1. Full high-match subgraph internals (deep JD analysis, cover letter, quality loop)
2. Human review interrupt workflow
3. Persistence checkpointer for pause/resume
4. Final ApplicationStrategy model output
