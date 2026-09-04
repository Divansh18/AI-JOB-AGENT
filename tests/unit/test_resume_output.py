from __future__ import annotations

from jobsearch.domain.resume_intelligence import (
    MasterResume,
    PAGE_STATUS_MISSING_OUTPUT,
    PAGE_STATUS_VALID,
    ResumeContentItem,
    ResumeEducationEntry,
    ResumeExperienceEntry,
    ResumeProjectEntry,
    ResumeSkillItem,
    ResumeSourceModel,
    ResumeTemplate,
    TailoredResume,
    TailoringRecommendation,
    validate_tailored_resume,
)
from jobsearch.services.resume_output import OutputLine, ResumeOutputState, _trim_output_state


def _source_model() -> ResumeSourceModel:
    return ResumeSourceModel(
        summary=ResumeContentItem(text="Software engineer building Python and FastAPI systems.", evidence_refs=["fact:1"]),
        skills=[
            ResumeSkillItem(skill="python", evidence_refs=["fact:2"]),
            ResumeSkillItem(skill="fastapi", evidence_refs=["fact:3"]),
            ResumeSkillItem(skill="typescript", evidence_refs=["fact:4"]),
        ],
        experience=[
            ResumeExperienceEntry(
                company="Acme Labs",
                title="Backend Engineering Intern",
                location="Bengaluru, India",
                start_date="2026-01-01",
                end_date="2026-06-30",
                summary=None,
                bullets=[
                    ResumeContentItem(text="Built Python APIs.", evidence_refs=["fact:5"]),
                    ResumeContentItem(text="Automated release checks.", evidence_refs=["fact:6"]),
                    ResumeContentItem(text="Improved internal tooling.", evidence_refs=["fact:7"]),
                ],
                evidence_refs=["fact:8"],
            )
        ],
        projects=[
            ResumeProjectEntry(
                name="API Copilot",
                role="Internal Developer Tool",
                link=None,
                summary=ResumeContentItem(text="Internal API review assistant.", evidence_refs=["fact:9"]),
                bullets=[
                    ResumeContentItem(text="Built the first release.", evidence_refs=["fact:10"]),
                    ResumeContentItem(text="Added verification workflows.", evidence_refs=["fact:11"]),
                ],
                skills=[ResumeSkillItem(skill="python", evidence_refs=["fact:12"])],
                evidence_refs=["fact:13"],
            ),
            ResumeProjectEntry(
                name="Playbook Search",
                role="Search Tool",
                link=None,
                summary=ResumeContentItem(text="Search over operational playbooks.", evidence_refs=["fact:14"]),
                bullets=[
                    ResumeContentItem(text="Indexed structured content.", evidence_refs=["fact:15"]),
                    ResumeContentItem(text="Added ranking controls.", evidence_refs=["fact:16"]),
                ],
                skills=[ResumeSkillItem(skill="typescript", evidence_refs=["fact:17"])],
                evidence_refs=["fact:18"],
            ),
        ],
        education=[
            ResumeEducationEntry(
                institution="Example University",
                degree="Bachelor of Computer Applications",
                field_of_study=None,
                start_date=None,
                end_date="2026-06-01",
                evidence_refs=["fact:19"],
            )
        ],
        links=[],
    )


def test_validate_tailored_resume_accepts_rendered_one_page_variant():
    source = _source_model()
    resume = TailoredResume(
        job_id=1,
        target_role="Software Engineer",
        master_resume=MasterResume(identity="master_v1", version=1, page_limit=1, source_status="ingested"),
        template=ResumeTemplate(
            identity="master_v1_template",
            master_resume_identity="master_v1",
            page_limit=1,
            layout_policy="preserve_existing_layout",
            render_validation_status=PAGE_STATUS_VALID,
        ),
        page_limit=1,
        page_validation_status=PAGE_STATUS_VALID,
        recommendation=TailoringRecommendation(
            decision="tailor",
            reasons=["Verified evidence can be reordered for the target role."],
            evidence_refs=["fact:5", "fact:10"],
        ),
        changes=[],
        preview=source,
        output_pdf_path="output/pdf/resume-job-1.pdf",
        page_count=1,
        evidence_refs_used=["fact:1", "fact:5", "fact:10"],
    )

    validation = validate_tailored_resume(resume, source)

    assert validation.ok is True


def test_validate_tailored_resume_rejects_missing_output_for_rendered_variant():
    source = _source_model()
    resume = TailoredResume(
        job_id=1,
        target_role="Software Engineer",
        master_resume=MasterResume(identity="master_v1", version=1, page_limit=1, source_status="ingested"),
        template=ResumeTemplate(
            identity="master_v1_template",
            master_resume_identity="master_v1",
            page_limit=1,
            layout_policy="preserve_existing_layout",
            render_validation_status=PAGE_STATUS_MISSING_OUTPUT,
        ),
        page_limit=1,
        page_validation_status=PAGE_STATUS_MISSING_OUTPUT,
        recommendation=TailoringRecommendation(
            decision="tailor",
            reasons=["Verified evidence can be reordered for the target role."],
            evidence_refs=["fact:5"],
        ),
        changes=[],
        preview=source,
        output_pdf_path=None,
        page_count=None,
        evidence_refs_used=["fact:5"],
    )

    validation = validate_tailored_resume(resume, source)
    codes = {issue.code for issue in validation.issues}

    assert validation.ok is False
    assert "missing_output_pdf_path" in codes


def test_trim_output_state_reduces_last_project_bullets_before_core_sections():
    source = _source_model()
    state = ResumeOutputState(
        full_name="Asha Example",
        contact_items=[OutputLine(text="asha@example.com", evidence_refs=["fact:20"])],
        section_order=["summary", "experience", "skills", "projects", "education"],
        summary=source.summary,
        skills=source.skills,
        experience=source.experience,
        projects=source.projects,
        certifications=[OutputLine(text="PCEP", evidence_refs=["fact:21"])],
        education=source.education,
    )

    trimmed = _trim_output_state(state)

    assert trimmed is not None
    assert len(trimmed.projects[-1].bullets) == 1
    assert len(trimmed.experience[0].bullets) == len(state.experience[0].bullets)
    assert trimmed.summary == state.summary
