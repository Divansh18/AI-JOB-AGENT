"""Deterministic signal extraction from job descriptions.

Every function here is pure and returns an explanation alongside its verdict,
so that a filter or ranking decision can always be traced to the exact phrase
that produced it.
"""

from __future__ import annotations

import re
import unicodedata
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
    EMEA_ONLY = "emea_only"
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

COUNTRY_NAMES = {
    "IN": "India",
    "US": "United States",
    "GB": "United Kingdom",
    "CA": "Canada",
    "MX": "Mexico",
    "TH": "Thailand",
    "CH": "Switzerland",
    "CN": "China",
    "DE": "Germany",
    "DK": "Denmark",
    "HU": "Hungary",
    "IL": "Israel",
    "IS": "Iceland",
    "IT": "Italy",
    "KR": "South Korea",
    "ES": "Spain",
    "IE": "Ireland",
    "EE": "Estonia",
    "PL": "Poland",
    "PT": "Portugal",
    "SA": "Saudi Arabia",
    "SG": "Singapore",
    "AU": "Australia",
    "AT": "Austria",
    "BE": "Belgium",
    "NL": "Netherlands",
    "FR": "France",
    "BR": "Brazil",
    "JP": "Japan",
    "AR": "Argentina",
    "AE": "United Arab Emirates",
    "SE": "Sweden",
    "UY": "Uruguay",
}

REGION_NAMES = {
    "EUROPE": "Europe",
    "EMEA": "EMEA",
    "APAC": "APAC",
}

COUNTRY_PATTERNS = [
    ("US", [
        r"\bunited states\b", r"\bu\.s\.a?\b", r"\busa\b", r"\bus-[a-z]{2}\b",
        r",\s*(?:ca|ny|wa|tx|ma|il|ga|co|nc|va)\b",
        r"\bsan francisco\b", r"\bnew york(?: city)?\b", r"\bmenlo park\b",
        r"\bsan diego\b", r"\bsan mateo\b", r"\bseattle\b", r"\baustin\b", r"\bboston\b",
        r"\bchicago\b", r"\batlanta\b", r"\bdenver\b", r"\blos angeles\b",
        r"\bbay area\b", r"\bnew jersey\b", r"\butah\b", r"\bsf office\b",
        r"\bny office\b", r"\bmountain view\b", r"\bwashington(?:,?\s*d\.?c\.?)\b",
    ]),
    ("GB", [r"\bunited kingdom\b", r"\buk\b", r"\blondon\b"]),
    ("CA", [r"\bcanada\b", r"\btoronto\b", r"\bvancouver\b", r"\bmontreal\b"]),
    ("MX", [r"\bmexico\b", r"\bmexico city\b"]),
    ("TH", [r"\bthailand\b", r"\bbangkok\b"]),
    ("CH", [r"\bswitzerland\b", r"\bzurich\b", r"\bgeneva\b"]),
    ("CN", [r"\bchina\b", r"\bbeijing\b", r"\bshanghai\b"]),
    ("DE", [r"\bgermany\b", r"\bberlin\b", r"\bmunich\b"]),
    ("DK", [r"\bdenmark\b", r"\bcopenhagen\b"]),
    ("HU", [r"\bhungary\b", r"\bbudapest\b"]),
    ("IL", [r"\bisrael\b", r"\btel aviv\b"]),
    ("IS", [r"\biceland\b", r"\breykjavik\b"]),
    ("IT", [r"\bitaly\b", r"\bmilan\b", r"\brome\b"]),
    ("KR", [r"\bkorea\b", r"\bsouth korea\b", r"\bseoul\b"]),
    ("ES", [r"\bspain\b", r"\bbarcelona\b", r"\bmadrid\b"]),
    ("IE", [r"\bireland\b", r"\bdublin\b"]),
    ("EE", [r"\bestonia\b", r"\btallinn\b"]),
    ("PL", [r"\bpoland\b", r"\bwarsaw\b", r"\bkrakow\b"]),
    ("PT", [r"\bportugal\b", r"\blisbon\b"]),
    ("SA", [r"\bsaudi arabia\b", r"\briyadh\b", r"\bjeddah\b"]),
    ("SG", [r"\bsingapore\b"]),
    ("AU", [r"\baustralia\b", r"\bsydney\b", r"\bmelbourne\b"]),
    ("AT", [r"\baustria\b", r"\bvienna\b"]),
    ("BE", [r"\bbelgium\b", r"\bbrussels\b"]),
    ("NL", [r"\bnetherlands\b", r"\bamsterdam\b"]),
    ("FR", [r"\bfrance\b", r"\bparis\b"]),
    ("BR", [r"\bbrazil\b", r"\bsao paulo\b"]),
    ("JP", [r"\bjapan\b", r"\btokyo\b"]),
    ("AR", [r"\bargentina\b", r"\bbuenos aires\b"]),
    ("AE", [r"\buae\b", r"\bunited arab emirates\b", r"\bdubai\b"]),
    ("SE", [r"\bsweden\b", r"\bstockholm\b"]),
    ("UY", [r"\buruguay\b", r"\bmontevideo\b"]),
]

REGION_PATTERNS = [
    ("EUROPE", [r"\beurope\b", r"\beu\b"]),
    ("EMEA", [r"\bemea\b"]),
    ("APAC", [r"\bapac\b", r"\basia[- ]pacific\b"]),
]

LOCATION_SNIPPET_PATTERNS = [
    r"\blocation\s*:\s*([^\n\r]{1,120})",
    r"\bbased in\s+([^\n\r]{1,120})",
    r"\bthis is (?:a|an)\s+(?:remote|hybrid|on-?site)\s+role(?:\s+(?:from|in|based in))?\s*([^\n\r]{0,120})",
    r"\bremote within\s+([^\n\r]{1,120})",
    r"\bwork from anywhere in\s+([^\n\r]{1,120})",
    r"\banywhere in india\b",
    r"\banywhere in the world\b",
]
DESCRIPTION_LOCATION_HEADER_CHARS = 700

DIRECT_HYBRID_PATTERNS = [r"\bhybrid\b"]
DIRECT_REMOTE_PATTERNS = [
    r"\bremote\b",
    r"\bwork from home\b",
    r"\bwfh\b",
    r"\bwork from anywhere\b",
    r"\banywhere in india\b",
    r"\banywhere in the world\b",
]
DIRECT_ONSITE_PATTERNS = [r"\bon-?site\b", r"\bin-?office\b", r"\bin person\b"]

DESCRIPTION_REMOTE_PATTERNS = [
    r"\bthis is (?:a|an)\s+remote role\b",
    r"\bthis role is remote\b",
    r"\bremote position\b",
    r"\blocation\s*:\s*remote\b",
    r"\bfully remote\b",
    r"\bremote within\b",
    r"\bwork from anywhere\b",
]
DESCRIPTION_HYBRID_PATTERNS = [
    r"\bthis is (?:a|an)\s+hybrid role\b",
    r"\bthis role is hybrid\b",
    r"\bhybrid position\b",
    r"\blocation\s*:\s*hybrid\b",
]
DESCRIPTION_ONSITE_PATTERNS = [
    r"\bthis is (?:a|an)\s+on-?site role\b",
    r"\bthis role is on-?site\b",
    r"\bon-?site position\b",
    r"\blocation\s*:\s*on-?site\b",
    r"\bin-?office\b",
    r"\bin person\b",
]

REMOTE_SCOPE_PATTERNS = [
    (RemoteScope.US_ONLY, [
        r"remote\s*[-–(,]?\s*(us|u\.s\.|usa|united states)",
        r"(us|u\.s\.|usa|united states)\s*[-–]\s*remote",
        r"(us|u\.s\.)\s+remote",
        r"(us|u\.s\.|usa|united states)[-\s]based (only|candidates|applicants)",
        r"must (reside|be located|live) in the (us|united states)",
        r"authorized to work in the (us|united states)",
        r"us work authorization",
        r"(us|u\.s\.) citizens? only",
        r"within the united states",
    ]),
    (RemoteScope.UK_ONLY, [r"remote\s*[-–(,]?\s*(uk|united kingdom)", r"right to work in the uk",
                           r"must (reside|be located) in the (uk|united kingdom)"]),
    (RemoteScope.EU_ONLY, [r"remote\s*[-–(,]?\s*(eu|europe)", r"must (reside|be located) in (the )?(eu|europe)",
                           r"eu work authorization", r"european union only"]),
    (RemoteScope.EMEA_ONLY, [r"remote\s*[-–(,]?\s*emea", r"\bemea\b"]),
    (RemoteScope.CANADA_ONLY, [r"remote\s*[-–(,]?\s*canada", r"must (reside|be located) in canada"]),
    (RemoteScope.INDIA, [r"remote\s*[-–(,]?\s*india", r"india\s*[-–(,]?\s*remote",
                         r"work from home.{0,20}india", r"anywhere in india"]),
    (RemoteScope.APAC, [r"remote\s*[-–(,]?\s*(apac|asia)", r"\bapac\b", r"asia[- ]pacific"]),
    (RemoteScope.GLOBAL, [r"remote\s*[-–(,]?\s*(global|worldwide|anywhere)",
                          r"work from anywhere",
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

def detect_work_auth_blocker(description: str) -> str | None:
    d = (description or "").lower()
    for phrase in WORK_AUTH_BLOCKERS:
        if phrase in d:
            return phrase
    return None


@dataclass(frozen=True)
class LocationFacts:
    cities: list[str]
    countries: list[str]
    regions: list[str]
    country: str | None
    remote_type: RemoteType
    remote_scope: RemoteScope
    multi_location: bool
    used_description_fallback: bool
    evidence: list[str] = field(default_factory=list)

    def as_signal(self) -> dict:
        return {
            "cities": self.cities,
            "countries": self.countries,
            "regions": self.regions,
            "country": self.country,
            "remote_type": self.remote_type.value,
            "remote_scope": self.remote_scope.value,
            "multi_location": self.multi_location,
            "used_description_fallback": self.used_description_fallback,
            "evidence": self.evidence,
        }


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _fold(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", (text or "").lower())
        if not unicodedata.combining(c)
    )


def _india_cities(text: str) -> list[str]:
    low = _fold(text)
    cities: list[str] = []
    for key, canonical in INDIA_CITIES.items():
        if re.search(rf"\b{re.escape(key)}\b", low) and canonical not in cities:
            cities.append(canonical)
    return cities


def _country_codes(text: str) -> list[str]:
    low = _fold(text)
    codes: list[str] = []
    if _india_cities(low) or re.search(r"\bindia\b", low):
        codes.append("IN")
    for code, patterns in COUNTRY_PATTERNS:
        if any(re.search(pattern, low) for pattern in patterns):
            codes.append(code)
    return _dedupe(codes)


def _region_codes(text: str) -> list[str]:
    low = _fold(text)
    codes: list[str] = []
    for code, patterns in REGION_PATTERNS:
        if any(re.search(pattern, low) for pattern in patterns):
            codes.append(code)
    return _dedupe(codes)


def _description_location_snippets(description: str) -> list[str]:
    low = _fold(description)
    snippets: list[str] = []
    for pattern in LOCATION_SNIPPET_PATTERNS:
        for match in re.finditer(pattern, low):
            snippet = match.group(0).strip()
            if snippet and snippet not in snippets:
                snippets.append(snippet[:120])
    for snippet in _header_location_snippets(description):
        if snippet and snippet not in snippets:
            snippets.append(snippet[:120])
    return snippets[:6]


def _header_location_snippets(description: str) -> list[str]:
    """Extract ATS header location metadata when location_raw was not provided."""
    header = _fold(description)[:DESCRIPTION_LOCATION_HEADER_CHARS]
    if not header:
        return []
    snippets: list[str] = []

    city_pattern = "|".join(re.escape(city) for city in sorted(INDIA_CITIES, key=len, reverse=True))
    direct_patterns = [
        rf"\b(?:{city_pattern})\s*,?\s*india\b(?:[^\n\r.]){{0,80}}",
        r"\bremote\s*[-–,/()]?\s*india\b(?:[^\n\r.]){0,80}",
        r"\bindia\s*[-–,/()]?\s*remote\b(?:[^\n\r.]){0,80}",
        r"\banywhere in india\b(?:[^\n\r.]){0,80}",
    ]
    for pattern in direct_patterns:
        for match in re.finditer(pattern, header):
            _add_header_snippet(snippets, match.group(0))

    for _, patterns in COUNTRY_PATTERNS:
        for pattern in patterns:
            for match in re.finditer(pattern, header):
                _add_header_snippet(
                    snippets,
                    header[max(0, match.start() - 40) : min(len(header), match.end() + 80)],
                )
    for _, patterns in REGION_PATTERNS:
        for pattern in patterns:
            for match in re.finditer(pattern, header):
                _add_header_snippet(
                    snippets,
                    header[max(0, match.start() - 40) : min(len(header), match.end() + 80)],
                )
    return snippets


def _add_header_snippet(snippets: list[str], value: str) -> None:
    snippet = _clean_header_snippet(value)
    if snippet and snippet not in snippets:
        snippets.append(snippet)


def _clean_header_snippet(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip(" -–—/,\t\r\n"))


def _location_is_generic(location: str) -> bool:
    return _fold(location).strip() in {"", "remote", "hybrid", "on-site", "onsite", "anywhere", "wfh"}


def detect_remote_type(title: str, location: str, description: str) -> RemoteType:
    direct = f"{title} {location}".lower()
    if any(re.search(p, direct) for p in DIRECT_HYBRID_PATTERNS):
        return RemoteType.HYBRID
    if any(re.search(p, direct) for p in DIRECT_REMOTE_PATTERNS):
        return RemoteType.REMOTE
    if any(re.search(p, direct) for p in DIRECT_ONSITE_PATTERNS):
        return RemoteType.ONSITE

    if _location_is_generic(location):
        header_snippets = " ".join(_header_location_snippets(description))
        if any(re.search(p, header_snippets) for p in DIRECT_HYBRID_PATTERNS):
            return RemoteType.HYBRID
        if any(re.search(p, header_snippets) for p in DIRECT_REMOTE_PATTERNS):
            return RemoteType.REMOTE
        if any(re.search(p, header_snippets) for p in DIRECT_ONSITE_PATTERNS):
            return RemoteType.ONSITE

    snippets = " ".join(_description_location_snippets(description))
    if any(re.search(p, snippets) for p in DESCRIPTION_HYBRID_PATTERNS):
        return RemoteType.HYBRID
    if any(re.search(p, snippets) for p in DESCRIPTION_REMOTE_PATTERNS):
        return RemoteType.REMOTE
    if any(re.search(p, snippets) for p in DESCRIPTION_ONSITE_PATTERNS):
        return RemoteType.ONSITE
    return RemoteType.UNKNOWN


def detect_remote_scope(location: str, description: str, title: str = "") -> tuple[RemoteScope, str | None]:
    blob = " ".join([f"{title} {location}".lower(), *_description_location_snippets(description)])
    for scope, patterns in REMOTE_SCOPE_PATTERNS:
        for p in patterns:
            m = re.search(p, blob)
            if m:
                return scope, m.group(0)
    return RemoteScope.UNKNOWN, None


def extract_location_facts(title: str, location_raw: str, description: str) -> LocationFacts:
    """Extract deterministic location facts from authoritative location text."""
    location_text = (location_raw or "").strip()
    cities = _india_cities(location_text)
    countries = _country_codes(location_text)
    regions = _region_codes(location_text)
    evidence = [location_text] if location_text else []
    used_description_fallback = False
    remote_type = detect_remote_type(title, location_raw, description)

    should_use_description = _location_is_generic(location_text)
    if remote_type in (RemoteType.REMOTE, RemoteType.HYBRID) and not countries and not regions:
        should_use_description = True

    if should_use_description and not cities and not countries and not regions:
        snippets = _description_location_snippets(description)
        if snippets:
            used_description_fallback = True
            for snippet in snippets:
                evidence.append(snippet)
                cities.extend(_india_cities(snippet))
                countries.extend(_country_codes(snippet))
                regions.extend(_region_codes(snippet))
            cities = _dedupe(cities)
            countries = _dedupe(countries)
            regions = _dedupe(regions)

    country = None
    if "IN" in countries:
        country = "IN"
    elif len(countries) == 1:
        country = countries[0]

    remote_scope, scope_evidence = detect_remote_scope(location_raw, description, title)
    if scope_evidence and scope_evidence not in evidence:
        evidence.append(scope_evidence[:120])

    low = location_text.lower()
    multi_location = len(countries) > 1 or len(cities) > 1 or len(regions) > 1
    if not multi_location and location_text:
        multi_location = bool(
            re.search(r"[;/]|(?:\b(?:and|or)\b)", low) and (countries or cities or regions)
        )

    return LocationFacts(
        cities=cities,
        countries=countries,
        regions=regions,
        country=country,
        remote_type=remote_type,
        remote_scope=remote_scope,
        multi_location=multi_location,
        used_description_fallback=used_description_fallback,
        evidence=evidence,
    )


def parse_locations(location_raw: str, description: str = "") -> tuple[list[str], str | None]:
    """Return (canonical India cities found, inferred primary country code)."""
    facts = extract_location_facts("", location_raw, description)
    return facts.cities, facts.country
