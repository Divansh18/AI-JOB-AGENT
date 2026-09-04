from __future__ import annotations

from jobsearch.domain.master_resume import (
    PdfLink,
    build_master_resume_facts,
    parse_master_resume_text,
)

SAMPLE_RESUME_TEXT = """Asha Example
asha@example.com | +91 99999 88888 | LinkedIn | GitHub
SUMMARY
Software engineer building Python and FastAPI applications for internal teams.
EXPERIENCE
Backend Engineering Intern | Acme Labs | Jan 2026 - Jun 2026
● Built Python and FastAPI APIs used by internal teams.
● Automated release checks with Playwright.
Software Engineer Intern | Beta Systems | Jul 2025 - Dec 2025
● Built export workflows using NestJS and TypeScript.
SKILLS
Languages: Python, TypeScript, SQL
Frontend: Next.js, React.js
Backend: FastAPI, NestJS
Testing: Playwright
PROJECTS
API Copilot — Internal Developer Tool — Python, FastAPI, PostgreSQL | GitHub
● Built a developer tool for API review.
Portfolio Tracker — TypeScript, Next.js | Live Demo | GitHub
● Built a market dashboard for personal finance.
CERTIFICATIONS
PCEP - Certified Entry-Level Python Programmer (Python Institute)
EDUCATION
Example University | Bachelor of Computer Applications (BCA) — Graduated: June 2026
"""


def _parsed():
    return parse_master_resume_text(
        SAMPLE_RESUME_TEXT,
        page_count=1,
        links=[
            PdfLink(label="LinkedIn", url="https://linkedin.com/in/asha-example", page=1, top=742.0),
            PdfLink(label="GitHub", url="https://github.com/asha-example", page=1, top=742.0),
            PdfLink(label="GitHub", url="https://github.com/asha-example/api-copilot", page=1, top=330.0),
        ],
    )


def test_parse_master_resume_text_detects_sections_and_preserves_structure():
    parsed = _parsed()

    assert parsed.page_count == 1
    assert parsed.full_name == "Asha Example"
    assert parsed.email == "asha@example.com"
    assert parsed.phone == "+91 99999 88888"
    assert parsed.profile_links["linkedin_url"] == "https://linkedin.com/in/asha-example"
    assert parsed.profile_links["github_url"] == "https://github.com/asha-example"
    assert parsed.sections_detected == [
        "summary",
        "experience",
        "skills",
        "projects",
        "certifications",
        "education",
    ]
    assert parsed.summary == "Software engineer building Python and FastAPI applications for internal teams."
    assert len(parsed.experience) == 2
    assert parsed.experience[0].company == "Acme Labs"
    assert parsed.experience[0].bullets[0] == "Built Python and FastAPI APIs used by internal teams."
    assert len(parsed.skills) == 8
    assert parsed.projects[0].name == "API Copilot"
    assert parsed.projects[0].tagline == "Internal Developer Tool"
    assert parsed.projects[0].tech_stack == ["Python", "FastAPI", "PostgreSQL"]
    assert parsed.certifications[0].name == "PCEP - Certified Entry-Level Python Programmer (Python Institute)"
    assert parsed.education[0].degree == "Bachelor of Computer Applications (BCA)"


def test_build_master_resume_facts_preserves_source_and_safe_skill_proficiency():
    parsed = _parsed()

    facts = build_master_resume_facts(parsed, source="master_resume_v1")

    assert facts
    assert all(fact.source == "master_resume_v1" for fact in facts)
    summary_fact = next(fact for fact in facts if fact.selector == "personal_profile:summary")
    python_fact = next(fact for fact in facts if fact.key == "skill_python")
    react_fact = next(fact for fact in facts if fact.key == "skill_react_js")
    experience_fact = next(fact for fact in facts if fact.category == "work_experience")
    certification_fact = next(fact for fact in facts if fact.key.startswith("certification_"))

    assert summary_fact.value == "Software engineer building Python and FastAPI applications for internal teams."
    assert python_fact.value["proficiency"] == "professional"
    assert python_fact.value["source_contexts"] == ["project", "skills", "work_experience"]
    assert react_fact.value["proficiency"] is None
    assert experience_fact.value["date_text"] == "Jan 2026 - Jun 2026"
    assert certification_fact.value["kind"] == "certification"


def test_parse_master_resume_text_keeps_wrapped_project_bullet_with_same_project():
    parsed = parse_master_resume_text(
        """Asha Example
asha@example.com | +91 99999 88888
PROJECTS
API Copilot — Internal Developer Tool — Python, FastAPI | GitHub
● Built and deployed an internal tool that automatically reviews API changes
in under 60 seconds without storing customer payloads.
● Added audit logging with rule-based scoring —
without splitting the current project.
""",
        page_count=1,
        links=[],
    )

    assert len(parsed.projects) == 1
    assert parsed.projects[0].name == "API Copilot"
    assert parsed.projects[0].bullets == [
        "Built and deployed an internal tool that automatically reviews API changes in under 60 seconds without storing customer payloads.",
        "Added audit logging with rule-based scoring — without splitting the current project.",
    ]
