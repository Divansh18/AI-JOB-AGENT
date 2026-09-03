"""Architectural boundaries.

These enforce two promises made in the Phase 0 spec:
  1. domain/ stays pure, and cli/ is a thin adapter, so a web dashboard can be
     added later without touching business logic.
  2. only the provider layer may invoke a model runtime.
"""

import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "jobsearch"


def _modules(package: str) -> list[Path]:
    return [p for p in (SRC / package).rglob("*.py") if p.name != "__init__.py"]


def _imports(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    return set(re.findall(r"^\s*(?:from|import)\s+([.\w]+)", text, re.M))


# --- layering --------------------------------------------------------------

def test_domain_has_no_io_dependencies():
    """domain/ must be pure: no db, no http, no filesystem, no subprocess."""
    banned = {"sqlite3", "httpx", "requests", "subprocess", "pathlib", "os"}
    for path in _modules("domain"):
        for imp in _imports(path):
            root = imp.split(".")[0]
            assert root not in banned, f"{path.name} imports {imp}"


def test_domain_does_not_import_upper_layers():
    for path in _modules("domain"):
        for imp in _imports(path):
            for layer in ("services", "persistence", "sources", "cli", "render", "llm"):
                assert f"{layer}." not in imp and not imp.endswith(f".{layer}"), \
                    f"{path.name} imports {imp}"


def test_services_do_not_import_cli_or_render():
    """A web UI must be able to call services without dragging in the CLI."""
    for path in _modules("services"):
        for imp in _imports(path):
            assert ".cli" not in imp, f"{path.name} imports {imp}"


def test_cli_contains_no_sql():
    """All SQL belongs in persistence/repositories.py."""
    for path in _modules("cli"):
        text = path.read_text(encoding="utf-8")
        for stmt in ("INSERT INTO", "UPDATE ", "DELETE FROM"):
            assert stmt not in text, f"{path.name} contains raw SQL: {stmt}"


def test_persistence_owns_the_schema():
    sql_files = list((SRC / "persistence" / "migrations").glob("*.sql"))
    assert sql_files, "no migrations found"
    for path in SRC.rglob("*.py"):
        if "persistence" in path.parts:
            continue
        assert "CREATE TABLE" not in path.read_text(encoding="utf-8"), path


# --- provider isolation ----------------------------------------------------

def test_only_claude_cli_module_can_execute_a_subprocess():
    """No module outside the provider layer may shell out to a model runtime.

    Config may *name* the binary (llm.cli_binary) - naming is not invoking.
    What is forbidden anywhere else is process execution.
    """
    allowed = {SRC / "llm" / "claude_cli.py"}
    exec_call = re.compile(
        r"subprocess\.(run|Popen|call|check_output|check_call)"
        r"|os\.(system|popen|exec\w*|spawn\w*)"
    )
    offenders = []
    for path in SRC.rglob("*.py"):
        if path in allowed:
            continue
        if exec_call.search(path.read_text(encoding="utf-8")):
            offenders.append(str(path.relative_to(SRC)))
    assert not offenders, f"modules executing a subprocess: {offenders}"


def test_binary_name_appears_only_in_config_and_provider():
    """The literal `claude` binary name is confined to two places."""
    allowed = {"llm/claude_cli.py", "config/schemas.py"}
    offenders = []
    for path in SRC.rglob("*.py"):
        rel = str(path.relative_to(SRC))
        if rel in allowed:
            continue
        if re.search(r'\bcli_binary\b|["\']claude["\']', path.read_text(encoding="utf-8")):
            offenders.append(rel)
    assert not offenders, f"binary name leaked into: {offenders}"


def test_no_module_outside_provider_layer_imports_a_concrete_provider():
    """Callers must go through llm.provider.get_provider()."""
    allowed = {"provider.py", "claude_cli.py", "anthropic_provider.py"}
    for path in SRC.rglob("*.py"):
        if path.parent.name == "llm" and path.name in allowed:
            continue
        for imp in _imports(path):
            assert "claude_cli" not in imp, f"{path.name} imports {imp}"
            assert "anthropic_provider" not in imp, f"{path.name} imports {imp}"


def test_anthropic_sdk_is_not_imported_at_module_load():
    """An absent optional dependency must never break the deterministic path."""
    text = (SRC / "llm" / "anthropic_provider.py").read_text(encoding="utf-8")
    module_level = [l for l in text.splitlines()
                    if l.startswith("import anthropic") or l.startswith("from anthropic")]
    assert not module_level, "anthropic must be imported lazily inside functions"


def test_deterministic_pipeline_does_not_depend_on_llm():
    """discover -> filter -> rank must work with the LLM layer absent."""
    for module in ("services/pipeline.py", "services/digest.py", "domain/ranking.py",
                   "domain/filters.py"):
        text = (SRC / module).read_text(encoding="utf-8")
        assert "llm" not in _imports(SRC / module), f"{module} depends on llm"


def test_triage_is_the_only_service_touching_llm():
    for path in _modules("services"):
        for imp in _imports(path):
            assert not imp.startswith("..llm"), f"{path.name} imports {imp}"


# --- safety ----------------------------------------------------------------

def test_no_hardcoded_years_of_experience_for_the_candidate():
    """Phase 0 must not assert a numeric experience figure anywhere."""
    profile = Path(__file__).resolve().parents[2] / "config" / "profile.yaml"
    text = profile.read_text(encoding="utf-8")
    assert "years_experience" not in text
    assert "experience_stage: early_career" in text


def test_profile_schema_rejects_a_years_field():
    from jobsearch.config.schemas import Profile
    assert "years_experience" not in Profile.model_fields
