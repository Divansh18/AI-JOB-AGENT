"""Audit logging helper."""

from __future__ import annotations

from ..persistence.repositories import EventRepo


class Audit:
    def __init__(self, conn):
        self.repo = EventRepo(conn)

    def system(self, action: str, **payload) -> None:
        self.repo.log("system", action, payload=payload)

    def human(self, action: str, entity_type: str | None = None,
              entity_id: str | int | None = None, **payload) -> None:
        self.repo.log("human", action, entity_type, entity_id, payload)

    def entity(self, actor: str, action: str, entity_type: str,
               entity_id: str | int, **payload) -> None:
        self.repo.log(actor, action, entity_type, entity_id, payload)
