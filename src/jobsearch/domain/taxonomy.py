"""Role, seniority and skill vocabularies.

Pure data + pure functions. No I/O, no config loading. Everything here is
deliberately explicit so that a classification decision can be traced back to
a specific token rather than to an opaque model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class TitleTier(str, Enum):
    """How well a job title matches the target role set."""

    CORE_SWE = "core_swe"
    STACK = "stack"
    AI = "ai"
    ADJACENT = "adjacent"
    AMBIGUOUS = "ambiguous"
    EXCLUDED = "excluded"


# --- Role vocabularies -----------------------------------------------------
# Matched against the normalized title (lowercase, punctuation collapsed).

CORE_SWE_PATTERNS = [
    r"\bsoftware engineer\b",
    r"\bsoftware developer\b",
    r"\bsoftware development engineer\b",
    r"\bsde\b",
    r"\bassociate software\b",
    r"\bgraduate software\b",
    r"\bgraduate engineer\b",
    r"\bjunior (software )?(engineer|developer)\b",
    r"\bentry level (software )?(engineer|developer)\b",
    r"\bnew grad\b",
    r"\bmember of technical staff\b",
    r"\bapplication (engineer|developer)\b",
    r"\bprogrammer analyst\b",
    r"\bproduct engineer\b",
]

STACK_PATTERNS = [
    r"\bfull.?stack\b",
    r"\bbackend\b",
    r"\bback.end\b",
    r"\bfrontend\b",
    r"\bfront.end\b",
    r"\bweb developer\b",
    r"\bweb engineer\b",
    r"\bapi engineer\b",
    r"\bplatform engineer\b",
    r"\bnode\.?js (engineer|developer)\b",
    r"\breact (engineer|developer)\b",
    r"\bpython (engineer|developer)\b",
    r"\bjavascript (engineer|developer)\b",
    r"\btypescript (engineer|developer)\b",
]

AI_PATTERNS = [
    r"\bai engineer\b",
    r"\bapplied ai\b",
    r"\bai/?ml engineer\b",
    r"\bml engineer\b",
    r"\bmachine learning engineer\b",
    r"\bllm engineer\b",
    r"\bgen.?ai engineer\b",
    r"\bai (software )?developer\b",
    r"\bai application\b",
    r"\bagent(ic)? engineer\b",
    r"\bforward deployed engineer\b",
]

ADJACENT_PATTERNS = [
    r"\bdata engineer\b",
    r"\bsolutions engineer\b",
    r"\bintegration engineer\b",
    r"\bsystems engineer\b",
    r"\bmobile (engineer|developer)\b",
    r"\bandroid (engineer|developer)\b",
    r"\bios (engineer|developer)\b",
]

# Titles we never want, regardless of anything else in the posting.
EXCLUDED_ROLE_PATTERNS = [
    r"\bdev.?ops\b",
    r"\bsite reliability\b",
    r"\bsre\b",
    r"\bqa\b",
    r"\bquality assurance\b",
    r"\btest engineer\b",
    r"\bautomation test\b",
    r"\bsdet\b",
    r"\btechnical support\b",
    r"\bcustomer (support|success)\b",
    r"\bsales\b",
    r"\baccount executive\b",
    r"\brecruit(er|ing)\b",
    r"\btalent acquisition\b",
    r"\bdata entry\b",
    r"\bsalesforce\b",
    r"\bsap\b",
    r"\bmainframe\b",
    r"\bcobol\b",
    r"\bsharepoint\b",
    r"\bnetwork engineer\b",
    r"\bsecurity analyst\b",
    r"\bbusiness analyst\b",
    r"\bproject manager\b",
    r"\bproduct manager\b",
    r"\bscrum master\b",
    r"\bdesigner\b",
    r"\bmarketing\b",
    r"\bfinance\b",
    r"\bhr\b",
    r"\bteacher\b",
    r"\binstructor\b",
    r"\bfaculty\b",
]

# --- Seniority -------------------------------------------------------------
# Per the approved spec: Senior/Staff/Principal/Lead/Manager => filtered out.

SENIORITY_EXCLUDE_PATTERNS = [
    (r"\bsenior\b", "senior"),
    (r"\bsr\.?\b", "senior"),
    # "Member of Technical Staff" is an ordinary IC title at many companies,
    # so "staff" only counts as seniority when it is not preceded by "technical".
    (r"(?<!technical )\bstaff\b", "staff"),
    (r"\bprincipal\b", "principal"),
    (r"\blead\b", "lead"),
    (r"\btech(nical)? lead\b", "lead"),
    (r"\bteam lead\b", "lead"),
    (r"\barchitect\b", "architect"),
    (r"\bmanager\b", "manager"),
    (r"\bdirector\b", "director"),
    (r"\bhead of\b", "head"),
    (r"\bvp\b", "vp"),
    (r"\bvice president\b", "vp"),
    (r"\bchief\b", "chief"),
    (r"\bdistinguished\b", "distinguished"),
    (r"\bfellow\b", "fellow"),
]

# Roman/arabic level markers that imply beyond-entry level. Deliberately
# narrow: only matched as standalone level tokens so that "Web3" or
# "Python 3 Developer" never trip them.
LEVEL_TOKEN_PATTERNS = [
    (r"(?:^|[\s\-,(])(?:ii|iii|iv|v)(?:$|[\s\-,)])", "level_roman"),
    (r"\b(?:sde|swe|engineer|developer)\s*[-\s]?(?:2|3|4|5)\b", "level_numeric"),
    (r"\b(?:level|l)\s?(?:3|4|5|6|7)\b", "level_numeric"),
]

# Positive early-career signals in a title.
EARLY_CAREER_PATTERNS = [
    r"\bjunior\b",
    r"\bassociate\b",
    r"\bgraduate\b",
    r"\bnew grad\b",
    r"\bentry.level\b",
    r"\bcampus\b",
    r"\bsde.?1\b",
    r"\bsde.?i\b",
    r"\btrainee\b",
    r"\bearly career\b",
    r"\bfresher\b",
]


@dataclass(frozen=True)
class TitleAnalysis:
    """Result of classifying a job title."""

    tier: TitleTier
    matched_pattern: str | None
    seniority_token: str | None
    level_token: str | None
    early_career_signal: bool

    @property
    def is_senior(self) -> bool:
        return self.seniority_token is not None or self.level_token is not None


def _first_match(patterns: list[str], text: str) -> str | None:
    for p in patterns:
        if re.search(p, text):
            return p
    return None


def classify_title(normalized_title: str) -> TitleAnalysis:
    """Classify a *normalized* job title into a tier plus seniority signals."""
    t = normalized_title

    seniority_token = None
    for pattern, label in SENIORITY_EXCLUDE_PATTERNS:
        if re.search(pattern, t):
            seniority_token = label
            break

    level_token = None
    for pattern, label in LEVEL_TOKEN_PATTERNS:
        if re.search(pattern, t):
            level_token = label
            break

    early = any(re.search(p, t) for p in EARLY_CAREER_PATTERNS)

    # An excluded role wins over everything: we do not want a "Senior QA" or a
    # "Junior QA" either.
    excluded = _first_match(EXCLUDED_ROLE_PATTERNS, t)
    if excluded:
        return TitleAnalysis(TitleTier.EXCLUDED, excluded, seniority_token, level_token, early)

    for patterns, tier in (
        (CORE_SWE_PATTERNS, TitleTier.CORE_SWE),
        (AI_PATTERNS, TitleTier.AI),
        (STACK_PATTERNS, TitleTier.STACK),
        (ADJACENT_PATTERNS, TitleTier.ADJACENT),
    ):
        m = _first_match(patterns, t)
        if m:
            return TitleAnalysis(tier, m, seniority_token, level_token, early)

    # Unrecognised but not clearly wrong: keep it, rank it low. Optimising
    # against false negatives is an explicit design goal.
    return TitleAnalysis(TitleTier.AMBIGUOUS, None, seniority_token, level_token, early)


# --- Skills ----------------------------------------------------------------
# Aliases map surface forms found in job descriptions onto the canonical skill
# names used in profile.yaml. Keys must be lowercase.

SKILL_ALIASES: dict[str, str] = {
    "js": "javascript",
    "es6": "javascript",
    "ecmascript": "javascript",
    "ts": "typescript",
    "reactjs": "react.js",
    "react": "react.js",
    "react js": "react.js",
    "nextjs": "next.js",
    "next js": "next.js",
    "nodejs": "node.js",
    "node": "node.js",
    "node js": "node.js",
    "expressjs": "express.js",
    "express": "express.js",
    "nest": "nestjs",
    "nest.js": "nestjs",
    "fast api": "fastapi",
    "postgres": "postgresql",
    "psql": "postgresql",
    "mongo": "mongodb",
    "rest": "rest apis",
    "restful": "rest apis",
    "restful apis": "rest apis",
    "rest api": "rest apis",
    "api development": "rest apis",
    "amazon web services": "aws",
    "elastic search": "elasticsearch",
    "elk": "elasticsearch",
    "oauth": "oauth 2.0",
    "oauth2": "oauth 2.0",
    "json web token": "jwt",
    "json web tokens": "jwt",
    "web socket": "websockets",
    "websocket": "websockets",
    "socket.io": "websockets",
    "large language model": "llm orchestration",
    "large language models": "llm orchestration",
    "llm": "llm orchestration",
    "llms": "llm orchestration",
    "langchain": "llm orchestration",
    "llamaindex": "llm orchestration",
    "rag": "llm orchestration",
    "model context protocol": "mcp",
    "git hub": "github",
    "containers": "docker",
    "containerisation": "docker",
    "containerization": "docker",
    "unix": "linux",
    "structured query language": "sql",
}


def canonical_skill(term: str) -> str:
    """Map a raw skill mention onto its canonical name."""
    t = term.strip().lower()
    return SKILL_ALIASES.get(t, t)
