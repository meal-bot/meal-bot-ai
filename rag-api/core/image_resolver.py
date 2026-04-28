"""
Recipe image URL resolver.
recipe_id 목록을 받아 MySQL에서 대표/썸네일 이미지 URL을 한 번에 조회한다.
"""

from __future__ import annotations

import pymysql.cursors

from core.db import mysql_connection


def _normalize_url(value: object) -> str | None:
    if value is None:
        return None

    url = str(value).strip()
    if not url:
        return None
    return url


def get_recipe_images(recipe_ids: list[int]) -> dict[int, dict]:
    """
    recipe_id 목록에 대한 이미지 URL을 반환한다.
    DB 오류가 발생하면 빈 dict를 반환한다.
    """
    if not recipe_ids:
        return {}

    unique_ids = list(dict.fromkeys(recipe_ids))
    placeholders = ", ".join(["%s"] * len(unique_ids))
    sql = (
        "SELECT rcp_seq, img_main, img_thumb "
        f"FROM recipe WHERE rcp_seq IN ({placeholders})"
    )

    try:
        with mysql_connection() as conn:
            with conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(sql, unique_ids)
                rows = cur.fetchall()
    except Exception:
        return {}

    return {
        int(row["rcp_seq"]): {
            "image_url": _normalize_url(row.get("img_main")),
            "thumbnail_url": _normalize_url(row.get("img_thumb")),
        }
        for row in rows
    }
