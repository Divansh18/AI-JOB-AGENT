"""Load and validate YAML configuration."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from .schemas import CompaniesFile, Filters, Profile, RankingConfig, Settings


def project_root() -> Path:
    """Resolve the project root from the JOBSEARCH_ROOT env var or the CWD."""
    import os

    env = os.environ.get("JOBSEARCH_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = Path.cwd().resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "config").is_dir() and (candidate / "src" / "jobsearch").is_dir():
            return candidate
    return here


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@dataclass
class AppConfig:
    root: Path
    settings: Settings
    profile: Profile
    filters: Filters
    ranking: RankingConfig
    companies: CompaniesFile

    @property
    def db_path(self) -> Path:
        return self.root / self.settings.db_path

    @property
    def digest_dir(self) -> Path:
        return self.root / self.settings.digest_dir


def load_config(root: Path | None = None) -> AppConfig:
    root = root or project_root()
    cfg = root / "config"
    return AppConfig(
        root=root,
        settings=Settings(**_read_yaml(cfg / "settings.yaml")),
        profile=Profile(**_read_yaml(cfg / "profile.yaml")),
        filters=Filters(**_read_yaml(cfg / "filters.yaml")),
        ranking=RankingConfig(**_read_yaml(cfg / "ranking.yaml")),
        companies=CompaniesFile(**_read_yaml(cfg / "companies.yaml")),
    )


@lru_cache(maxsize=1)
def cached_config() -> AppConfig:
    return load_config()
