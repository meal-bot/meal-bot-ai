"""레시피 원본 JSON을 메모리에 적재하고 rcp_seq로 조회하는 인메모리 스토어."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class RecipeStore:
    """recipes_enriched_v2.json을 로드해 rcp_seq 키로 단건/배치 조회를 제공한다."""

    def __init__(self, json_path: str | Path) -> None:
        """JSON 파일을 로드해 {rcp_seq: recipe} 매핑으로 보관한다."""
        path = Path(json_path)
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        self._recipes: dict[str, dict] = {}
        for item in raw:
            if "rcp_seq" not in item:
                logger.warning("Skip item without rcp_seq: %s", item.get("name", "<unnamed>"))
                continue
            key = str(item["rcp_seq"])
            if key in self._recipes:
                logger.warning("Duplicate rcp_seq=%s, overwriting with later value", key)
            self._recipes[key] = item

        logger.info("RecipeStore loaded %d recipes from %s", len(self._recipes), path)

    def normalize_recipe_id(self, recipe_id: str) -> str:
        """'recipe_42' / '42' 형태를 모두 '42'로 정규화한다."""
        return str(recipe_id).removeprefix("recipe_")

    def get_recipe_by_id(self, recipe_id: str) -> dict | None:
        """단건 조회. 없으면 None."""
        normalized = self.normalize_recipe_id(recipe_id)
        hit = self._recipes.get(normalized)
        if hit is None:
            logger.debug("Recipe not found: %s (normalized=%s)", recipe_id, normalized)
        return hit

    def get_recipes_by_ids(self, recipe_ids: list[str]) -> list[dict]:
        """배치 조회. 입력 순서 보존, 없는 ID는 결과에서 제외."""
        result: list[dict] = []
        missing: list[str] = []
        for rid in recipe_ids:
            normalized = self.normalize_recipe_id(rid)
            hit = self._recipes.get(normalized)
            if hit is None:
                missing.append(normalized)
            else:
                result.append(hit)

        if missing:
            logger.warning(
                "Missing recipe_ids in batch lookup: requested=%d, found=%d, missing=%s",
                len(recipe_ids),
                len(result),
                missing,
            )
        return result

    def has_recipe(self, recipe_id: str) -> bool:
        """해당 recipe_id가 스토어에 존재하는지 확인."""
        return self.normalize_recipe_id(recipe_id) in self._recipes

    def __len__(self) -> int:
        return len(self._recipes)
