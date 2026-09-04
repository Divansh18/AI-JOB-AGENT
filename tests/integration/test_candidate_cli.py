import json
from pathlib import Path

from typer.testing import CliRunner

from jobsearch.cli.main import app

RUNNER = CliRunner()
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "candidate_import_v1.json"


def _write_config(root: Path) -> None:
    cfg = root / "config"
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "settings.yaml").write_text("db_path: data/test.db\ndigest_dir: data/digests\n", encoding="utf-8")
    (cfg / "profile.yaml").write_text("experience_stage: early_career\n", encoding="utf-8")
    (cfg / "filters.yaml").write_text("version: 2\n", encoding="utf-8")
    (cfg / "ranking.yaml").write_text("version: 2\n", encoding="utf-8")
    (cfg / "companies.yaml").write_text("version: 1\ncompanies: []\n", encoding="utf-8")


def test_candidate_cli_import_show_and_validate_excludes_unverified_fields(tmp_path):
    _write_config(tmp_path)
    env = {"JOBSEARCH_ROOT": str(tmp_path)}

    imported = RUNNER.invoke(app, ["candidate", "import-json", str(FIXTURE)], env=env)
    assert imported.exit_code == 0, imported.stdout

    shown = RUNNER.invoke(app, ["candidate", "show", "--json"], env=env)
    assert shown.exit_code == 0, shown.stdout
    payload = json.loads(shown.stdout)

    assert payload["profile"]["full_name"] == "Asha Example"
    assert payload["profile"]["email"] == "asha@example.com"
    assert payload["profile"]["phone"] is None
    assert payload["profile"]["portfolio_url"] is None
    assert payload["profile"]["preferred_locations"] == ["Bengaluru", "Pune", "Remote India"]
    assert payload["validation"]["ok"] is True

    validated = RUNNER.invoke(app, ["candidate", "validate", "--json"], env=env)
    assert validated.exit_code == 0, validated.stdout
    assert json.loads(validated.stdout)["ok"] is True


def test_answers_cli_forces_human_review_on_sensitive_keys(tmp_path):
    _write_config(tmp_path)
    env = {"JOBSEARCH_ROOT": str(tmp_path)}

    added = RUNNER.invoke(
        app,
        [
            "answers",
            "add",
            "work_authorization",
            "Authorized to work in India.",
            "--category",
            "attestation",
            "--verified",
            "--no-human-review",
        ],
        env=env,
    )
    assert added.exit_code == 0, added.stdout

    listed = RUNNER.invoke(app, ["answers", "list", "--json"], env=env)
    assert listed.exit_code == 0, listed.stdout
    answers = json.loads(listed.stdout)["answers"]

    assert len(answers) == 1
    assert answers[0]["question_key"] == "work_authorization"
    assert answers[0]["human_review_required"] is True
