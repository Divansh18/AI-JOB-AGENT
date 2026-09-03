"""Deterministic signal extraction from job descriptions.

Every function here is pure and returns an explanation alongside its verdict,
so that a filter or ranking decision can always be traced to the exact phrase
that produced it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

# ---------------------------------------------------------------------------
# Years of experience
# ---------------------------------------------------------------------------


class YoeVerdict(str, Enum):
    """Graded experience verdict.

    Deliberately graded rather than binary: the approved policy is to keep
    ambitious roles and penalise them in ranking, not to drop them.
    """

    UNKNOWN = "unknown"          # nothing stated -> keep, no adjustment
    STRONG = "strong"            # 0-2 required -> keep, bonus
    OK = "ok"                    # <=3 preferred -> keep, no penalty
    PENALTY = "penalty"          # 3 required -> keep, ranking penalty
    SOFT_HIGH = "soft_high"      # 4+ but only preferred -> keep, big penalty
    HARD_HIGH = "hard_high"      # 4+ explicitly required -> filter out


NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

_NUM = r"(?:\d{1,2}|zero|one|two|three|four|five|six|seven|eight|nine|ten)"

# Phrases that mean "this number is NOT a years-of-experience requirement".
YOE_NEGATIVE_GUARDS = [
    r"\d\s*year (degree|program|course|diploma|bachelor|master|college|university)",
    r"(within|in the (last|past)|over the (last|past))\s+\d+\s*year",
    r"\d+\s*year[- ]old",
    r"(founded|established|since)\s+\d{4}",
    r"\d+\s*years? (ago|of growth|in business|of operation)",
    r"next\s+\d+\s*years?",
    r"\d+\s*years?\s+in a row",
    r"(grown|growing|profitable|remote)\s+(for\s+)?\d+\s*years?",
    r"\d+\s*year (visa|contract|bond|vesting|cliff)",
]

# Ordered: the first matching pattern wins for a given text span.
YOE_RANGE_PATTERNS = [
    # "2-4 years", "2 to 4 years", "2 – 4 yrs"
    rf"({_NUM})\s*(?:-|–|—|to)\s*({_NUM})\s*\+?\s*(?:years?|yrs?)",
]

YOE_SINGLE_PATTERNS = [
    rf"({_NUM})\s*\+\s*(?:years?|yrs?)",
    rf"(?:at least|minimum(?: of)?|min\.?|no less than|over)\s+({_NUM})\s*(?:years?|yrs?)",
    rf"({_NUM})\s*(?:or more)\s*(?:years?|yrs?)",
    rf"({_NUM})\s*(?:years?|yrs?)",
]

REQUIRED_CUES = [
    "required", "requirement", "must have", "must possess", "minimum",
    "at least", "you have", "you bring", "we require", "mandatory",
    "should have", "need", "qualification",
]

PREFERRED_CUES = [
    "preferred", "preferable", "nice to have", "nice-to-have", "ideally",
    "bonus", "a plus", "plus point", "desirable", "good to have",
    "advantage", "would be great", "we'd love", "we would love",
]

FRESHER_CUES = [
    "fresher", "freshers", "no prior experience", "no experience required",
    "entry level", "entry-level", "new grad", "recent graduate",
    "final year student", "0-1 year", "0-2 year", "campus hire",
]


@dataclass(frozen=True)
class YoeMention:
    min_years: int
    max_years: int | None
    modality: str          # required | preferred | unknown
    snippet: str


@dataclass(frozen=True)
class YoeAnalysis:
    verdict: YoeVerdict
    required_min: int | None
    preferred_min: int | None
    mentions: list[YoeMention] = field(default_factory=list)
    evidence: str | None = None

    def as_signal(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "required_min": self.required_min,
            "preferred_min": self.preferred_min,
            "evidence": self.evidence,
        }


def _to_int(token: str) -> int | None:
    token = token.strip().lower()
    if token.isdigit():
        v = int(token)
        return v if 0 <= v <= 30 else None
    return NUMBER_WORDS.get(token)


def _modality_for(context: str) -> str:
    """Classify a surrounding text window as required / preferred / unknown."""
    c = context.lower()
    pref_hit = any(cue in c for cue in PREFERRED_CUES)
    req_hit = any(cue in c for cue in REQUIRED_CUES)
    if pref_hit and not req_hit:
        return "preferred"
    if pref_hit and req_hit:
        # "Required: 3+ years. Preferred: 5+ years" style. The closer cue wins.
        pref_pos = min((c.find(x) for x in PREFERRED_CUES if x in c), default=10**6)
        req_pos = min((c.find(x) for x in REQUIRED_CUES if x in c), default=10**6)
        return "preferred" if pref_pos < req_pos else "required"
    if req_hit:
        return "required"
    return "unknown"


def extract_yoe(text: str, *, window: int = 90) -> YoeAnalysis:
    """Extract years-of-experience requirements and grade them.

    Grading policy (approved):
        0-2 required -> STRONG      3 required   -> PENALTY
        <=3 preferred -> OK         4+ preferred -> SOFT_HIGH
        4+ required  -> HARD_HIGH   nothing      -> UNKNOWN
    """
    if not text:
        return YoeAnalysis(YoeVerdict.UNKNOWN, None, None)

    low = text.lower()
    mentions: list[YoeMention] = []
    consumed: list[tuple[int, int]] = []

    def overlaps(a: int, b: int) -> bool:
        return any(not (b <= s or a >= e) for s, e in consumed)

    def guarded(a: int, b: int) -> bool:
        ctx = low[max(0, a - 40) : min(len(low), b + 40)]
        return any(re.search(g, ctx) for g in YOE_NEGATIVE_GUARDS)

    for pattern in YOE_RANGE_PATTERNS:
        for m in re.finditer(pattern, low):
            if overlaps(*m.span()) or guarded(*m.span()):
                continue
            lo, hi = _to_int(m.group(1)), _to_int(m.group(2))
            if lo is None:
                continue
            ctx = low[max(0, m.start() - window) : m.end() + window]
            consumed.append(m.span())
            mentions.append(YoeMention(lo, hi, _modality_for(ctx), m.group(0).strip()))

    for pattern in YOE_SINGLE_PATTERNS:
        for m in re.finditer(pattern, low):
            if overlaps(*m.span()) or guarded(*m.span()):
                continue
            val = _to_int(m.group(1))
            if val is None:
                continue
            ctx = low[max(0, m.start() - window) : m.end() + window]
            consumed.append(m.span())
            mentions.append(YoeMention(val, None, _modality_for(ctx), m.group(0).strip()))

    fresher = any(cue in low for cue in FRESHER_CUES)

    if not mentions:
        if fresher:
            return YoeAnalysis(YoeVerdict.STRONG, 0, None, [], "early-career cue in description")
        return YoeAnalysis(YoeVerdict.UNKNOWN, None, None, [], None)

    # An unqualified mention is treated as "required": job descriptions that
    # state a number without hedging generally mean it.
    req = [m for m in mentions if m.modality in ("required", "unknown")]
    pref = [m for m in mentions if m.modality == "preferred"]

    required_min = min((m.min_years for m in req), default=None)
    preferred_min = min((m.min_years for m in pref), default=None)

    if required_min is not None:
        if required_min <= 2:
            verdict = YoeVerdict.STRONG
        elif required_min == 3:
            verdict = YoeVerdict.PENALTY
        else:
            verdict = YoeVerdict.HARD_HIGH
    elif preferred_min is not None:
        verdict = YoeVerdict.OK if preferred_min <= 3 else YoeVerdict.SOFT_HIGH
    else:
        verdict = YoeVerdict.UNKNOWN

    # A explicit fresher cue overrides a high requirement elsewhere in the
    # text (common in JDs that list several role levels in one posting).
    if fresher and verdict in (YoeVerdict.HARD_HIGH, YoeVerdict.SOFT_HIGH):
        verdict = YoeVerdict.PENALTY

    evidence = "; ".join(f"{m.snippet} [{m.modality}]" for m in mentions[:4])
    return YoeAnalysis(verdict, required_min, preferred_min, mentions, evidence)


# ---------------------------------------------------------------------------
# Employment type
# ---------------------------------------------------------------------------


class EmploymentType(str, Enum):
    FULL_TIME = "full_time"
    INTERNSHIP = "internship"
    CONTRACT = "contract"
    PART_TIME = "part_time"
    UNKNOWN = "unknown"


EMPLOYMENT_PATTERNS = [
    (EmploymentType.INTERNSHIP, [r"\bintern(ship)?\b", r"\bsummer analyst\b", r"\bco-?op\b", r"\bapprentice(ship)?\b"]),
    (EmploymentType.CONTRACT, [r"\bcontract(or)?\b", r"\bfreelance\b", r"\bc2h\b", r"\bcontract to hire\b",
                               r"\bconsultant\b", r"\btemporary\b", r"\bfixed.term\b", r"\bstaffing\b"]),
    (EmploymentType.PART_TIME, [r"\bpart.?time\b"]),
    (EmploymentType.FULL_TIME, [r"\bfull.?time\b", r"\bpermanent\b", r"\bregular\b"]),
]


def extract_employment_type(title: str, description: str, source_hint: str | None = None) -> EmploymentType:
    """Detect employment type. Title outweighs description; hint is last."""
    t = (title or "").lower()
    for etype, patterns in EMPLOYMENT_PATTERNS:
        if any(re.search(p, t) for p in patterns):
            return etype
    if source_hint:
        h = source_hint.lower().replace("-", " ").replace("_", " ")
        for etype, patterns in EMPLOYMENT_PATTERNS:
            if any(re.search(p, h) for p in patterns):
                return etype
    d = (description or "")[:3000].lower()
    for etype, patterns in EMPLOYMENT_PATTERNS:
        if any(re.search(p, d) for p in patterns):
            return etype
    return EmploymentType.UNKNOWN


# ---------------------------------------------------------------------------
# Remote scope and geography
# ---------------------------------------------------------------------------


class RemoteType(str, Enum):
    ONSITE = "onsite"
    HYBRID = "hybrid"
    REMOTE = "remote"
    UNKNOWN = "unknown"


class RemoteScope(str, Enum):
    INDIA = "india"
    APAC = "apac"
    GLOBAL = "global"
    US_ONLY = "us_only"
    UK_ONLY = "uk_only"
    EU_ONLY = "eu_only"
    CANADA_ONLY = "canada_only"
    OTHER_RESTRICTED = "other_restricted"
    UNKNOWN = "unknown"


INDIA_CITIES = {
    "bengaluru": "Bengaluru", "bangalore": "Bengaluru", "blr": "Bengaluru",
    "hyderabad": "Hyderabad", "secunderabad": "Hyderabad",
    "pune": "Pune", "mumbai": "Mumbai", "bombay": "Mumbai", "navi mumbai": "Mumbai",
    "thane": "Mumbai", "chennai": "Chennai", "madras": "Chennai",
    "delhi": "Delhi NCR", "new delhi": "Delhi NCR", "noida": "Delhi NCR",
    "gurugram": "Delhi NCR", "gurgaon": "Delhi NCR", "ncr": "Delhi NCR",
    "faridabad": "Delhi NCR", "ghaziabad": "Delhi NCR",
    "kolkata": "Kolkata", "ahmedabad": "Ahmedabad", "jaipur": "Jaipur",
    "indore": "Indore", "kochi": "Kochi", "cochin": "Kochi",
    "coimbatore": "Coimbatore", "chandigarh": "Chandigarh", "mohali": "Chandigarh",
    "trivandrum": "Thiruvananthapuram", "thiruvananthapuram": "Thiruvananthapuram",
    "bhubaneswar": "Bhubaneswar", "nagpur": "Nagpur", "vadodara": "Vadodara",
    "surat": "Surat", "lucknow": "Lucknow", "vizag": "Visakhapatnam",
    "visakhapatnam": "Visakhapatnam", "mysore": "Mysuru", "mysuru": "Mysuru",
}

US_STATE_HINTS = [
    " ca", " ny", " wa", " tx", " ma", " il", " ga", " co", " nc", " va",
    "california", "new york", "san francisco", "seattle", "austin", "boston",
    "chicago", "atlanta", "denver", "los angeles", "united states", "usa",
    " u.s.", "remote - us", "remote (us", "bay area", "new jersey", "utah",
]

REMOTE_SCOPE_PATTERNS = [
    (RemoteScope.US_ONLY, [
        r"remote\s*[-–(,]?\s*(us|u\.s\.|usa|united states)",
        r"(us|u\.s\.|usa|united states)[-\s]based (only|candidates|applicants)",
        r"must (reside|be located|live) in the (us|united states)",
        r"authorized to work in the (us|united states)",
        r"us work authorization",
        r"(us|u\.s\.) citizens? only",
        r"within the united states",
    ]),
    (RemoteScope.UK_ONLY, [r"remote\s*[-–(,]?\s*(uk|united kingdom)", r"right to work in the uk",
                           r"must (reside|be located) in the (uk|united kingdom)"]),
    (RemoteScope.EU_ONLY, [r"remote\s*[-–(,]?\s*(eu|europe|emea)", r"must (reside|be located) in (the )?(eu|europe)",
                           r"eu work authorization", r"european union only"]),
    (RemoteScope.CANADA_ONLY, [r"remote\s*[-–(,]?\s*canada", r"must (reside|be located) in canada"]),
    (RemoteScope.INDIA, [r"remote\s*[-–(,]?\s*india", r"india\s*[-–(,]?\s*remote",
                         r"work from home.{0,20}india", r"anywhere in india"]),
    (RemoteScope.APAC, [r"remote\s*[-–(,]?\s*(apac|asia)", r"asia[- ]pacific"]),
    (RemoteScope.GLOBAL, [r"remote\s*[-–(,]?\s*(global|worldwide|anywhere)",
                          r"work from anywhere", r"fully distributed",
                          r"anywhere in the world", r"globally remote"]),
]

WORK_AUTH_BLOCKERS = [
    "must be authorized to work in the united states",
    "must be legally authorized to work in the united states",
    "us citizens only",
    "u.s. citizens only",
    "security clearance",
    "ts/sci",
    "requires uk right to work",
    "green card holder",
    "no visa sponsorship",
    "not able to sponsor",
    "unable to provide sponsorship",
    "cannot provide visa sponsorship",
    "must have canadian citizenship",
    "eu citizenship required",
]


def detect_remote_type(title: str, location: str, description: str) -> RemoteType:
    blob = f"{title} {location} {description[:2500]}".lower()
    if re.search(r"\bhybrid\b", blob):
        return RemoteType.HYBRID
    if re.search(r"\b(remote|work from home|wfh|distributed|anywhere)\b", blob):
        return RemoteType.REMOTE
    if re.search(r"\b(on-?site|in-?office|in person)\b", blob):
        return RemoteType.ONSITE
    return RemoteType.UNKNOWN


def detect_remote_scope(location: str, description: str) -> tuple[RemoteScope, str | None]:
    blob = f"{location} {description[:4000]}".lower()
    for scope, patterns in REMOTE_SCOPE_PATTERNS:
        for p in patterns:
            m = re.search(p, blob)
            if m:
                return scope, m.group(0)
    return RemoteScope.UNKNOWN, None


def detect_work_auth_blocker(description: str) -> str | None:
    d = (description or "").lower()
    for phrase in WORK_AUTH_BLOCKERS:
        if phrase in d:
            return phrase
    return None


def parse_locations(location_raw: str) -> tuple[list[str], str | None]:
    """Return (canonical india cities found, inferred country code)."""
    if not location_raw:
        return [], None
    l = location_raw.lower()
    cities: list[str] = []
    for key, canonical in INDIA_CITIES.items():
        if re.search(rf"\b{re.escape(key)}\b", l) and canonical not in cities:
            cities.append(canonical)
    if cities or re.search(r"\bindia\b", l):
        return cities, "IN"
    if any(h in l for h in US_STATE_HINTS):
        return [], "US"
    for token, code in [("united kingdom", "GB"), ("london", "GB"), ("canada", "CA"),
                        ("germany", "DE"), ("berlin", "DE"), ("singapore", "SG"),
                        ("australia", "AU"), ("ireland", "IE"), ("dublin", "IE"),
                        ("netherlands", "NL"), ("amsterdam", "NL"), ("france", "FR"),
                        ("poland", "PL"), ("spain", "ES"), ("brazil", "BR"),
                        ("japan", "JP"), ("uae", "AE"), ("dubai", "AE")]:
        if re.search(rf"\b{re.escape(token)}\b", l):
            return [], code
    return [], None
