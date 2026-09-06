"""Validated configuration models.

Every YAML file the system reads is parsed into one of these. Validation
failures are loud and specific: a bad config should never silently degrade
into bad ranking.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


# --- profile.yaml ----------------------------------------------------------


class SkillSet(BaseModel):
    strong: list[str] = Field(default_factory=list)
    familiar: list[str] = Field(default_factory=list)

    @field_validator("strong", "familiar")
    @classmethod
    def _lower(cls, v: list[str]) -> list[str]:
        return [s.strip().lower() for s in v if s.strip()]


class Profile(BaseModel):
    """Phase 0 profile: preferences and skills only.

    Deliberately contains NO numeric years-of-experience field and no
    employment claims. Experience is described qualitatively as a stage, and
    the verified Truth Store (Phase 1) will hold exact dates and evidence.
    Nothing in this file is ever used to generate text about the candidate.
    """

    experience_stage: str = "early_career"
    target_roles: list[str] = Field(default_factory=list)
    skills: SkillSet = Field(default_factory=SkillSet)
    preferred_locations: list[str] = Field(default_factory=list)
    remote_ok: bool = True
    summary_for_matching: str = ""

    @field_validator("experience_stage")
    @classmethod
    def _stage(cls, v: str) -> str:
        allowed = {"early_career", "mid", "senior"}
        if v not in allowed:
            raise ValueError(f"experience_stage must be one of {sorted(allowed)}")
        return v

    def matching_text(self) -> str:
        """Text embedded for semantic similarity."""
        parts = [self.summary_for_matching.strip()]
        if self.target_roles:
            parts.append("Target roles: " + ", ".join(self.target_roles) + ".")
        if self.skills.strong:
            parts.append("Core skills: " + ", ".join(self.skills.strong) + ".")
        if self.skills.familiar:
            parts.append("Developing familiarity with: " + ", ".join(self.skills.familiar) + ".")
        return " ".join(p for p in parts if p)


# --- filters.yaml ----------------------------------------------------------


class TitleFilters(BaseModel):
    include_tiers: list[str] = Field(default_factory=lambda: ["core_swe", "stack", "ai"])
    keep_ambiguous: bool = True
    keep_adjacent: bool = True
    exclude_senior_titles: bool = True


class ExperienceFilters(BaseModel):
    exclude_verdicts: list[str] = Field(default_factory=lambda: ["hard_high"])
    unknown_policy: str = "pass"


class RemoteFilters(BaseModel):
    allow: bool = True
    allowed_scopes: list[str] = Field(
        default_factory=lambda: ["india", "apac", "global"]
    )
    exclude_scopes: list[str] = Field(
        default_factory=lambda: [
            "us_only", "uk_only", "eu_only", "emea_only", "canada_only", "other_restricted",
        ]
    )


class LocationFilters(BaseModel):
    preferred_cities: list[str] = Field(default_factory=list)
    allow_countries: list[str] = Field(default_factory=lambda: ["IN"])
    remote: RemoteFilters = Field(default_factory=RemoteFilters)
    unknown_policy: str = "pass"


class EmploymentFilters(BaseModel):
    preferred: list[str] = Field(default_factory=lambda: ["full_time"])
    deprioritize: list[str] = Field(default_factory=lambda: ["internship"])
    exclude: list[str] = Field(default_factory=lambda: ["contract", "part_time"])


class CompanyFilters(BaseModel):
    blocklist: list[str] = Field(default_factory=list)
    # Staffing agencies repost the same roles across many clients and rarely
    # lead anywhere. Defaults mirror config/filters.yaml so an absent config
    # file still behaves sensibly.
    blocklist_patterns: list[str] = Field(
        default_factory=lambda: [
            "staffing", "recruiters", "manpower", "consultancy services",
            "talent solutions", "hr services",
        ]
    )


class ContentFilters(BaseModel):
    exclude_phrases: list[str] = Field(default_factory=list)
    min_description_chars: int = 120


class FreshnessFilters(BaseModel):
    max_age_days: int = 45


class WorkAuthFilters(BaseModel):
    exclude_blocked: bool = True


class Filters(BaseModel):
    version: int = 2
    titles: TitleFilters = Field(default_factory=TitleFilters)
    experience: ExperienceFilters = Field(default_factory=ExperienceFilters)
    locations: LocationFilters = Field(default_factory=LocationFilters)
    employment: EmploymentFilters = Field(default_factory=EmploymentFilters)
    companies: CompanyFilters = Field(default_factory=CompanyFilters)
    content: ContentFilters = Field(default_factory=ContentFilters)
    freshness: FreshnessFilters = Field(default_factory=FreshnessFilters)
    work_authorization: WorkAuthFilters = Field(default_factory=WorkAuthFilters)


# --- ranking.yaml ----------------------------------------------------------


class Weights(BaseModel):
    semantic: float = 40.0
    title: float = 20.0
    freshness: float = 15.0
    location: float = 10.0
    skills: float = 10.0
    priority: float = 5.0


class YoeAdjustments(BaseModel):
    strong: float = 4.0
    ok: float = 0.0
    unknown: float = 0.0
    penalty: float = -8.0
    soft_high: float = -15.0


class EmploymentAdjustments(BaseModel):
    full_time: float = 0.0
    internship: float = -12.0
    unknown: float = -2.0
    part_time: float = -10.0


class RankingConfig(BaseModel):
    version: int = 2
    weights: Weights = Field(default_factory=Weights)
    yoe_adjustments: YoeAdjustments = Field(default_factory=YoeAdjustments)
    employment_adjustments: EmploymentAdjustments = Field(default_factory=EmploymentAdjustments)
    early_career_title_bonus: float = 5.0
    freshness_halflife_hours: float = 96.0
    familiar_skill_weight: float = 0.35
    semantic_percentile_normalize: bool = True


# --- companies.yaml --------------------------------------------------------


class CompanyEntry(BaseModel):
    name: str
    ats: str
    token: str
    priority: str = "normal"
    tags: list[str] = Field(default_factory=list)
    hq_country: str | None = None
    active: bool = True
    verified_at: str | None = None
    notes: str | None = None

    @field_validator("ats")
    @classmethod
    def _ats(cls, v: str) -> str:
        allowed = {"greenhouse", "lever", "ashby"}
        if v not in allowed:
            raise ValueError(f"ats must be one of {sorted(allowed)}")
        return v

    @field_validator("priority")
    @classmethod
    def _prio(cls, v: str) -> str:
        allowed = {"high", "normal", "low"}
        if v not in allowed:
            raise ValueError(f"priority must be one of {sorted(allowed)}")
        return v


class CompaniesFile(BaseModel):
    version: int = 1
    companies: list[CompanyEntry] = Field(default_factory=list)


# --- settings.yaml ---------------------------------------------------------


class HttpSettings(BaseModel):
    timeout_seconds: float = 20.0
    max_retries: int = 3
    requests_per_second: float = 1.0
    user_agent: str = "jobsearch-personal/0.1 (personal job search agent)"


class LlmSettings(BaseModel):
    """LLM configuration.

    Provider selection is config-driven: switching between the temporary
    Claude Code CLI runtime, Anthropic API, and OpenAI API requires only a
    change to this value plus the relevant environment variable. No
    application code changes. The API key is never a config field - it comes from the
    environment only.
    """

    enabled: bool = False
    provider: str = "claude_cli"
    model: str = "claude-haiku-4-5"
    daily_cap_inr: float = 15.0
    monthly_cap_inr: float = 500.0
    per_call_cap_inr: float = 25.0
    usd_to_inr: float = 88.0
    max_jd_chars: int = 6000
    max_input_chars: int = 16000
    max_output_tokens: int = 1024
    prompt_version: str = "application_intelligence_v1"
    answer_prompt_version: str = "application_answer_drafting_v1"
    resume_wording_prompt_version: str = "resume_wording_v2"
    # claude_cli provider only
    cli_binary: str = "claude"
    cli_timeout_seconds: int = 120
    # API providers only: anthropic and openai
    api_timeout_seconds: int = 60

    @field_validator("provider")
    @classmethod
    def _provider(cls, v: str) -> str:
        allowed = {"claude_cli", "anthropic", "openai"}
        if v not in allowed:
            raise ValueError(f"llm.provider must be one of {sorted(allowed)}")
        return v


class DigestSettings(BaseModel):
    top_n: int = 20
    aging_min_score: float = 70.0
    aging_days: int = 10
    followup_days: int = 7


class Settings(BaseModel):
    timezone: str = "Asia/Kolkata"
    db_path: str = "data/jobsearch.db"
    digest_dir: str = "data/digests"
    http: HttpSettings = Field(default_factory=HttpSettings)
    llm: LlmSettings = Field(default_factory=LlmSettings)
    digest: DigestSettings = Field(default_factory=DigestSettings)
