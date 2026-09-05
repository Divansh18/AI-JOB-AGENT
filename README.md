# jobsearch — Phase 0

Personal job discovery, ranking and application ledger.

**Scope:** `discover → normalize → dedupe → hard-filter → rank → daily digest → application ledger`

Phase 0 runs with **zero recurring AI cost**. Embeddings run locally on CPU; the
LLM layer is present but disabled.

## Quick start

```bash
.venv/bin/jobsearch init          # create db, run migrations
.venv/bin/jobsearch doctor        # validate config, report health

# build the company list (probes public ATS boards; nothing is guessed)
.venv/bin/jobsearch companies discover --names-file config/seed_names.txt
.venv/bin/jobsearch companies review
.venv/bin/jobsearch companies sync

# the daily loop
.venv/bin/jobsearch pipeline run
.venv/bin/jobsearch digest --open
```

Daily cron:

```cron
30 8 * * * cd ~/ai-job-agent && .venv/bin/jobsearch pipeline run && .venv/bin/jobsearch digest
```

## The daily loop

```
jobsearch pipeline run              # fetch, dedupe, filter, rank
jobsearch digest --open             # read the digest
jobsearch jobs show <id>            # full JD + score breakdown
jobsearch apply plan <id>           # create an attended application plan
jobsearch apply record <id>         # record a completed application
jobsearch status <id> recruiter_reply
jobsearch stats                     # funnel + response rate
```

## Architecture

```
cli/  render/     thin adapters — parse args, call a service, format output
   ↓
services/         use cases. A web dashboard would call exactly these.
   ↓
domain/           pure logic: no I/O, no db, no network, no subprocess
persistence/      all SQL lives here
sources/          one adapter per job source
llm/              optional, disabled by default
```

Boundaries are enforced by `lint-imports` (4 contracts) and by
`tests/unit/test_architecture.py`. `domain/` importing `httpx` or `sqlite3`
fails the build.

## Configuration

| File | Purpose |
|---|---|
| `config/profile.yaml` | Preferences and skills. **No numeric experience value**, no employment claims. |
| `config/filters.yaml` | Hard filters. Every rule has a stable id. |
| `config/ranking.yaml` | Scoring weights. Re-ranking is free. |
| `config/companies.yaml` | Verified target companies. |
| `config/settings.yaml` | Paths, HTTP politeness, LLM provider. |
| `config/seed_names.txt` | Company **names** for discovery. Never slugs. |

### Experience policy

Nothing asserts a number of years for the candidate. `profile.yaml` records
`experience_stage: early_career` only. Exact employment dates and evidence
belong in the verified Truth Store (Phase 1).

Job-side experience requirements are **graded, not binary**:

| Job says | Verdict | Effect |
|---|---|---|
| 0–2 years required | `strong` | kept, **+4** ranking bonus |
| ≤3 years preferred | `ok` | kept, no change |
| nothing stated | `unknown` | **kept**, no change |
| 3 years required | `penalty` | kept, **−8** |
| 4+ years preferred | `soft_high` | kept, **−15** |
| 4+ years required | `hard_high` | **excluded** |
| Senior/Staff/Principal/Lead/Manager title | — | **excluded** |

Filters are tuned against **false negatives**: wherever a signal is missing,
the job is kept. Missing a good role costs more than reading a bad card.

## Job sources

| Source | Auth | Notes |
|---|---|---|
| Greenhouse | none | public board API, per company |
| Lever | none | public postings API |
| Ashby | none | public posting API |
| Manual | — | `jobsearch jobs add <url>` |
| Adzuna | free key | **optional**; skipped unless `ADZUNA_APP_ID`/`ADZUNA_APP_KEY` are set |

**Coverage caveat:** these three ATSs skew toward startups, product companies
and remote-friendly global employers. Many Indian mid-size and services
companies use Naukri, Darwinbox, Keka or Workday and **will not appear**.
Closing that gap is Phase 1 (job-alert emails via Gmail).

### Company discovery never guesses

An ATS token only reaches `companies.yaml` after: HTTP 200 + parsable payload
+ at least one live posting + a name-similarity check against the input name.
Candidates land in `companies.candidates.yaml` and need your keystroke to be
accepted. Slug-guessing has a low hit rate, which is precisely why every hit
is verified.

## LLM layer (optional, disabled)

Phase 0 uses **no LLM**. `llm.enabled: false` and `jobsearch triage` refuses to
run. Provider selection is config-driven:

```yaml
llm:
  enabled: false
  provider: claude_cli    # temporary local runtime (needs `claude` on PATH)
  # provider: anthropic   # long-term (needs ANTHROPIC_API_KEY in the env)
```

Switching providers is a config + env change — no application code changes.

Guarantees enforced by tests:
- Only `llm/claude_cli.py` may execute a subprocess.
- Both providers validate through a single `validate_into()` path, so their
  outputs cannot drift apart.
- `ANTHROPIC_API_KEY` is read from the environment only. It is never a config
  field, never logged, and never forwarded to the CLI subprocess.
- Timeouts, non-zero exits, malformed JSON and a missing binary all map onto
  the same `LLMError` hierarchy.
- Discovery, filtering and ranking never import `llm`. If the LLM is
  unavailable, the pipeline is unaffected.

Estimated cost if enabled (Haiku 4.5, $1/$5 per MTok, ₹88/USD):

| Setting | ₹/month |
|---|---|
| off (default) | **₹0** |
| top 15/day | ≈₹137 |
| top 25/day | ≈₹228 |

## Testing

```bash
.venv/bin/python -m pytest tests/ -q     # 181 tests, no network
.venv/bin/lint-imports                   # architecture contracts
```

Golden fixtures are recorded API payloads, so an ATS changing its response
shape fails a test rather than silently producing empty jobs.
# AI-JOB-AGENT
