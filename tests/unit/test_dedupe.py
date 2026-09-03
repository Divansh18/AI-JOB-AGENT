from datetime import datetime, timedelta, timezone

from jobsearch.domain.dedupe import find_duplicates
from jobsearch.domain.fingerprint import canonicalize_url, content_hash, identity_fingerprint
from tests.conftest import make_job

NOW = datetime.now(timezone.utc)


def test_identity_fingerprint_is_stable_and_distinct():
    a = dict(source="greenhouse", external_id="gh:1", company_normalized="acme",
             title_normalized="software engineer", location_normalized="pune")
    assert identity_fingerprint(**a) == identity_fingerprint(**a)
    assert identity_fingerprint(**{**a, "external_id": "gh:2"}) != identity_fingerprint(**a)


def test_fingerprint_falls_back_to_company_title_location():
    """Without an external id, the same role from two sources collapses."""
    a = identity_fingerprint(source="adzuna", external_id=None, company_normalized="acme",
                             title_normalized="backend engineer", location_normalized="pune")
    b = identity_fingerprint(source="manual", external_id=None, company_normalized="acme",
                             title_normalized="backend engineer", location_normalized="pune")
    assert a == b


def test_canonicalize_url_strips_tracking_and_case():
    assert (canonicalize_url("https://WWW.Example.com/jobs/9/?utm_source=x#a")
            == "https://example.com/jobs/9")
    assert canonicalize_url(None) is None


def test_content_hash_detects_edits():
    a = content_hash("Engineer", "We use Python", "Pune")
    assert a == content_hash("engineer", "we use python  ", "pune")
    assert a != content_hash("Engineer", "We use Go", "Pune")


def test_cross_source_duplicate_is_linked_to_earliest():
    early = make_job(fingerprint="a", external_id="gh:1", source="greenhouse",
                     title="Backend Engineer", first_seen_at=NOW - timedelta(hours=5),
                     canonical_url="https://a.example/1")
    late = make_job(fingerprint="b", external_id="az:9", source="adzuna",
                    title="Backend Engineer", first_seen_at=NOW,
                    canonical_url="https://b.example/9")
    links = find_duplicates([late, early])
    assert len(links) == 1
    assert links[0].canonical_fingerprint == "a", "earliest first_seen must be canonical"
    assert links[0].duplicate_fingerprint == "b"


def test_same_canonical_url_is_a_duplicate():
    a = make_job(fingerprint="a", external_id="gh:1", title="Software Engineer",
                 canonical_url="https://x.example/1", first_seen_at=NOW - timedelta(hours=1))
    b = make_job(fingerprint="b", external_id="lv:2", title="Software Engineer II",
                 canonical_url="https://x.example/1", first_seen_at=NOW)
    links = find_duplicates([a, b])
    assert len(links) == 1 and links[0].method == "canonical_url"


def test_different_companies_are_not_duplicates():
    a = make_job(fingerprint="a", external_id="1", company_normalized="acme",
                 canonical_url="https://a.example/1")
    b = make_job(fingerprint="b", external_id="2", company_normalized="globex",
                 canonical_url="https://b.example/2")
    assert find_duplicates([a, b]) == []


def test_different_roles_at_same_company_are_not_duplicates():
    a = make_job(fingerprint="a", external_id="1", title="Backend Engineer",
                 canonical_url="https://a.example/1")
    b = make_job(fingerprint="b", external_id="2", title="Frontend Engineer",
                 canonical_url="https://a.example/2")
    assert find_duplicates([a, b]) == []


def test_dedupe_is_deterministic_and_idempotent():
    jobs = [
        make_job(fingerprint="a", external_id="1", title="Software Engineer",
                 first_seen_at=NOW - timedelta(hours=2), canonical_url="https://a/1"),
        make_job(fingerprint="b", external_id="2", title="Software Engineer",
                 first_seen_at=NOW, canonical_url="https://a/2"),
    ]
    assert find_duplicates(jobs) == find_duplicates(list(reversed(jobs)))


# --- regression: location-aware fuzzy dedupe --------------------------------

def test_same_title_in_different_cities_is_not_a_duplicate():
    """A large employer posts one title in many cities. Merging them would
    hide an India opening behind a foreign canonical."""
    a = make_job(fingerprint="a", external_id="1", title="Account Manager",
                 location_raw="Mexico City", locations=[], country=None,
                 canonical_url="https://x/1", first_seen_at=NOW - timedelta(hours=1))
    b = make_job(fingerprint="b", external_id="2", title="Account Manager",
                 location_raw="Singapore", locations=[], country=None,
                 canonical_url="https://x/2", first_seen_at=NOW)
    assert find_duplicates([a, b]) == []


def test_india_posting_is_never_merged_into_a_foreign_canonical():
    foreign = make_job(fingerprint="a", external_id="1", title="Software Engineer",
                       location_raw="San Francisco, CA", locations=[], country="US",
                       canonical_url="https://x/1", first_seen_at=NOW - timedelta(hours=2))
    india = make_job(fingerprint="b", external_id="2", title="Software Engineer",
                     location_raw="Bengaluru, India", locations=["Bengaluru"],
                     country="IN", canonical_url="https://x/2", first_seen_at=NOW)
    assert find_duplicates([foreign, india]) == []


def test_same_city_different_source_still_deduplicates():
    """The legitimate cross-source case must keep working."""
    gh = make_job(fingerprint="a", external_id="gh:1", source="greenhouse",
                  title="Backend Engineer", location_raw="Bengaluru, India",
                  locations=["Bengaluru"], country="IN",
                  canonical_url="https://gh/1", first_seen_at=NOW - timedelta(hours=3))
    az = make_job(fingerprint="b", external_id="az:1", source="adzuna",
                  title="Backend Engineer", location_raw="Bangalore",
                  locations=["Bengaluru"], country="IN",
                  canonical_url="https://az/1", first_seen_at=NOW)
    links = find_duplicates([gh, az])
    assert len(links) == 1 and links[0].canonical_fingerprint == "a"


def test_remote_variants_still_deduplicate():
    a = make_job(fingerprint="a", external_id="1", title="Backend Engineer",
                 location_raw="Remote", locations=[], country=None,
                 canonical_url="https://x/1", first_seen_at=NOW - timedelta(hours=1))
    b = make_job(fingerprint="b", external_id="2", title="Backend Engineer",
                 location_raw="Remote - India", locations=[], country=None,
                 canonical_url="https://x/2", first_seen_at=NOW)
    assert len(find_duplicates([a, b])) == 1


def test_two_indian_cities_are_separate_requisitions():
    a = make_job(fingerprint="a", external_id="1", title="Software Engineer",
                 location_raw="Bengaluru, India", locations=["Bengaluru"], country="IN",
                 canonical_url="https://x/1", first_seen_at=NOW - timedelta(hours=1))
    b = make_job(fingerprint="b", external_id="2", title="Software Engineer",
                 location_raw="Hyderabad, India", locations=["Hyderabad"], country="IN",
                 canonical_url="https://x/2", first_seen_at=NOW)
    assert find_duplicates([a, b]) == []
