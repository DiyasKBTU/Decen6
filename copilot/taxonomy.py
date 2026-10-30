"""Таксономия категорий и приоритетов из taxonomy.json."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

from . import config


@dataclass(frozen=True)
class Taxonomy:
    categories: dict[str, dict]
    priorities: list[str]
    needs_human_always: frozenset[str]

    def rules_of(self, category: str) -> list[str]:
        return self.categories.get(category, {}).get("rules", [])

    def category_for_ref(self, ref: str, kind: str) -> str | None:
        """Пункт правил -> категория, в чьём списке он стоит («GZ» = любой пункт GZ)."""
        for code, cat in self.categories.items():
            if cat["kind"] != kind:
                continue
            for r in cat["rules"]:
                if r == ref or (r == "GZ" and ref.startswith("GZ ")) or ref.startswith(r + "."):
                    return code
        return None


@lru_cache(maxsize=1)
def get_taxonomy() -> Taxonomy:
    data = json.loads(config.TAXONOMY_PATH.read_text(encoding="utf-8"))
    return Taxonomy(
        categories={c["code"]: c for c in data["categories"]},
        priorities=[p["code"] for p in data["priorities"]],
        needs_human_always=frozenset(data.get("needs_human_always", [])),
    )
