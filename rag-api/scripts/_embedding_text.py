"""
임베딩 텍스트 생성 공통 모듈.
preview_embedding.py, build_vector_db.py 에서 import.
"""
from __future__ import annotations

import re


def clean_ingredients(text: str) -> str:
    if not text:
        return ""
    t = text.replace('\n', ' ')
    t = re.sub(r'\([^)]*\)', ' ', t)
    t = re.sub(r'\[[^\]]*\]', ' ', t)
    t = re.sub(r'[½⅓⅔¼¾⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞]', ' ', t)
    t = re.sub(r'\d+(?:[.\/]\d+)?\s*(?:g|kg|ml|l|L|개|마리|봉지|작은술|큰술|컵|대|쪽|장|모|T|t|cm|줌|꼬집)?', ' ', t)
    t = re.sub(r'양념장\s*[:：]|양념\s*[:：]|재료\s*[:：]', ' ', t)
    t = re.sub(r'[●•○■□◆◇▪▫·‧・]', ' ', t)
    t = re.sub(r'[,:：]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def build_embedding_text(recipe: dict) -> str:
    parts = [recipe['name']]
    cat = recipe.get('category')
    if cat and cat != '기타':
        parts.append(f"{cat.replace('&', ' ')} 요리")
    way = recipe.get('cooking_way')
    if way and way != '기타':
        parts.append(f"{way} 방식")
    ing = clean_ingredients(recipe.get('ingredients', ''))
    if ing:
        parts.append(f"주재료 {ing}")
    tag = recipe.get('hash_tag')
    if tag:
        parts.append(f"특징 {tag}")
    return ". ".join(parts) + "."