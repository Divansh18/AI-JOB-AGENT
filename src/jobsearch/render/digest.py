"""Render DigestData to a self-contained HTML file."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

TEMPLATE_DIR = Path(__file__).parent / "templates"


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        # Autoescape unconditionally. select_autoescape() matches on file
        # extension, and this template is ".j2" - which silently left escaping
        # OFF. Job titles and company names come from third-party APIs, so
        # unescaped rendering is a real injection vector in a file opened in
        # the browser.
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_html(data) -> str:
    return _env().get_template("digest.html.j2").render(d=data)


def write_digest(data, digest_dir: Path) -> Path:
    digest_dir.mkdir(parents=True, exist_ok=True)
    path = digest_dir / f"{data.digest_date}.html"
    path.write_text(render_html(data), encoding="utf-8")
    return path
