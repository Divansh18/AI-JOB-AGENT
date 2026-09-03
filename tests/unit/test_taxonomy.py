import pytest

from jobsearch.domain.normalize import normalize_title
from jobsearch.domain.taxonomy import TitleTier, canonical_skill, classify_title

TITLE_CASES = [
    ("Software Engineer", TitleTier.CORE_SWE, False),
    ("Software Development Engineer I", TitleTier.CORE_SWE, False),
    ("SDE 1", TitleTier.CORE_SWE, False),
    ("SDE-1, Backend", TitleTier.CORE_SWE, False),
    ("Associate Software Engineer", TitleTier.CORE_SWE, False),
    ("Graduate Software Engineer", TitleTier.CORE_SWE, False),
    ("Software Engineer - New Grad", TitleTier.CORE_SWE, False),
    ("Member of Technical Staff", TitleTier.CORE_SWE, False),
    ("Junior Developer", TitleTier.CORE_SWE, False),
    ("Full Stack Engineer", TitleTier.STACK, False),
    ("Full-Stack Developer", TitleTier.STACK, False),
    ("Backend Engineer", TitleTier.STACK, False),
    ("Front-End Developer", TitleTier.STACK, False),
    ("Node.js Developer", TitleTier.STACK, False),
    ("AI Engineer", TitleTier.AI, False),
    ("Applied AI Engineer", TitleTier.AI, False),
    ("Machine Learning Engineer", TitleTier.AI, False),
    ("Forward Deployed Engineer", TitleTier.AI, False),
    ("Data Engineer", TitleTier.ADJACENT, False),
    ("SDK Engineer - JavaScript", TitleTier.STACK, False),
    ("Associate Infrastructure Engineer", TitleTier.ADJACENT, False),
    ("Performance Engineer - Benchmarking", TitleTier.ADJACENT, False),
    # senior variants: tier is preserved, seniority flag is what excludes
    ("Senior Software Engineer", TitleTier.CORE_SWE, True),
    ("Sr. Backend Engineer", TitleTier.STACK, True),
    ("Staff Engineer", TitleTier.AMBIGUOUS, True),
    ("Principal Software Engineer", TitleTier.CORE_SWE, True),
    ("Engineering Manager", TitleTier.AMBIGUOUS, True),
    ("Tech Lead", TitleTier.AMBIGUOUS, True),
    ("Software Engineer II", TitleTier.CORE_SWE, True),
    ("Backend Engineer III", TitleTier.STACK, True),
    ("Director of Engineering", TitleTier.AMBIGUOUS, True),
    ("Head of Platform", TitleTier.AMBIGUOUS, True),
    # never-target roles
    ("QA Engineer", TitleTier.EXCLUDED, False),
    ("SDET", TitleTier.EXCLUDED, False),
    ("DevOps Engineer", TitleTier.EXCLUDED, False),
    ("Site Reliability Engineer", TitleTier.EXCLUDED, False),
    ("Salesforce Developer", TitleTier.EXCLUDED, False),
    ("Product Manager", TitleTier.EXCLUDED, True),
    ("Technical Recruiter", TitleTier.EXCLUDED, False),
    ("Business Analyst", TitleTier.EXCLUDED, False),
    ("Intermediate Support Engineer", TitleTier.EXCLUDED, False),
    ("Customer Experience Engineer, L1", TitleTier.EXCLUDED, False),
    ("Developer Relations Engineer", TitleTier.EXCLUDED, False),
    ("AI Field Engineer - Enterprise", TitleTier.EXCLUDED, False),
    ("Value Solutions Engineer (Inside Presales - America Region)", TitleTier.EXCLUDED, False),
    ("Solution Engineering", TitleTier.EXCLUDED, False),
    ("Solution Engineer - Insurance & Asset Management", TitleTier.EXCLUDED, False),
    ("Service Desk Specialist", TitleTier.EXCLUDED, False),
    ("Strategy and Operations Associate", TitleTier.EXCLUDED, False),
    ("Associate - Monetisation", TitleTier.EXCLUDED, False),
    ("Accounts Payable, Spend Management Coordinator", TitleTier.EXCLUDED, False),
    ("Analyst (Supply Analytics, Bangkok-based, Relocation provided)", TitleTier.EXCLUDED, False),
    ("PPSL- Product Management- Devices", TitleTier.EXCLUDED, False),
    ("Deal Desk", TitleTier.EXCLUDED, False),
    ("Intern - Admin and Operations", TitleTier.EXCLUDED, False),
    ("Product Research Specialist (W from Groww)", TitleTier.EXCLUDED, False),
    ("Language Expert - Taiwan (Bangkok Based)", TitleTier.EXCLUDED, False),
    ("Creative Sourcer", TitleTier.EXCLUDED, False),
    ("growth and business - max and wallet", TitleTier.EXCLUDED, False),
    ("Workplace & Engagement Coordinator", TitleTier.EXCLUDED, False),
]


@pytest.mark.parametrize("title,tier,is_senior", TITLE_CASES, ids=[c[0] for c in TITLE_CASES])
def test_classify_title(title, tier, is_senior):
    a = classify_title(normalize_title(title))
    assert a.tier == tier, f"{title!r} -> {a.tier}"
    assert a.is_senior == is_senior, f"{title!r} seniority -> {a.seniority_token}/{a.level_token}"


def test_level_tokens_do_not_false_positive():
    """Version numbers and tech names must not read as seniority levels."""
    for title in ["Web3 Engineer", "Python 3 Developer", "Vue 3 Frontend Engineer",
                  "React Native Engineer", "AI/ML Engineer"]:
        a = classify_title(normalize_title(title))
        assert not a.is_senior, f"{title!r} wrongly flagged senior ({a.level_token})"


def test_early_career_signals():
    for title in ["Graduate Engineer", "Associate Software Engineer", "SDE 1",
                  "Software Engineer - New Grad", "Junior Backend Developer"]:
        assert classify_title(normalize_title(title)).early_career_signal, title


def test_engineering_titles_with_business_context_are_not_excluded():
    for title, tier in [
        ("Product Engineer - Manufacturing Operations", TitleTier.CORE_SWE),
        ("Software Engineer, Monetization", TitleTier.CORE_SWE),
        ("Software Engineer, Growth", TitleTier.CORE_SWE),
        ("Software Engineer, Analytics Platform", TitleTier.CORE_SWE),
        ("Research Software Engineer", TitleTier.CORE_SWE),
        ("Associate Infrastructure Engineer", TitleTier.ADJACENT),
        ("Business Systems Developer", TitleTier.AMBIGUOUS),
    ]:
        analysis = classify_title(normalize_title(title))
        assert analysis.tier == tier, title


@pytest.mark.parametrize("raw,canonical", [
    ("JS", "javascript"), ("ReactJS", "react.js"), ("Node", "node.js"),
    ("Postgres", "postgresql"), ("RESTful APIs", "rest apis"),
    ("LangChain", "llm orchestration"), ("OAuth2", "oauth 2.0"),
    ("Model Context Protocol", "mcp"), ("Elastic Search", "elasticsearch"),
])
def test_skill_aliases(raw, canonical):
    assert canonical_skill(raw) == canonical
