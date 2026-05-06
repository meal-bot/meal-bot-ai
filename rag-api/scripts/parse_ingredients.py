"""recipes.json의 ingredients raw 텍스트를 구조화하는 최소 파서."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import OrderedDict
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SRC_PATH = DATA_DIR / "recipes.json"

UNICODE_FRACTIONS = {
    "⅓": "1/3", "⅔": "2/3",
    "¼": "1/4", "½": "1/2", "¾": "3/4",
    "⅛": "1/8", "⅜": "3/8", "⅝": "5/8", "⅞": "7/8",
    "⅕": "1/5", "⅖": "2/5", "⅗": "3/5", "⅘": "4/5",
    "⅙": "1/6", "⅚": "5/6",
}

HTML_BR_RE = re.compile(r"<\s*br\s*/?\s*>", re.IGNORECASE)
BULLET_RE = re.compile(r"[·•※●○◆◇■□▶►]")
WS_RE = re.compile(r"[ \t]+")
MULTI_NL_RE = re.compile(r"\n{2,}")
LINE_PREFIX_PORTION_RE = re.compile(r"^\s*\[\s*(\d+\s*인분|간편식\s*재료|추가\s*재료|메인\s*재료|기본\s*재료|재료)\s*\]\s*")
MIXED_FRACTION_RE = re.compile(r"(\d)([⅓⅔¼½¾⅛⅜⅝⅞⅕⅖⅗⅘⅙⅚])")

SAUCE_KEYWORDS = ("양념장", "양념", "소스", "드레싱")
GARNISH_KEYWORDS = ("고명", "토핑", "장식")

NON_NUMERIC_AMOUNTS = ("약간", "적당량", "소량", "조금")

HEADER_RE = re.compile(r"^\s*[\[\(]?\s*([^\[\]\(\):]+?)\s*[\]\)]?\s*[:：]?\s*$")
SECTION_INLINE_RE = re.compile(r"^\s*\[?([^\[\]\:：]+?)\]?\s*[:：]\s*(.+)$")

PORTION_HEADERS_RE = re.compile(
    r"^(\d+\s*인분|\d+인분\s*기준|간편식\s*재료|추가\s*재료|메인\s*재료|기본\s*재료|재료)$"
)


def normalize_text(raw: str, recipe_name: str, flags: dict) -> str:
    """1단계: HTML/유니코드/머리표 정규화."""
    if HTML_BR_RE.search(raw):
        flags["had_html"] = True
    text = HTML_BR_RE.sub("\n", raw)

    if any(ch in text for ch in UNICODE_FRACTIONS):
        flags["had_unicode_fraction"] = True
    # 숫자 + 유니코드 분수 (mixed number) 사이에 공백 삽입: "1⅓" → "1 1/3"
    text = MIXED_FRACTION_RE.sub(r"\1 \2", text)
    for ch, repl in UNICODE_FRACTIONS.items():
        text = text.replace(ch, repl)

    text = unicodedata.normalize("NFC", text)

    lines = text.split("\n")
    if lines:
        first = lines[0].strip()
        # 머리표 제거 후 비교 (●방울토마토 소박이 같은 케이스)
        first_stripped = BULLET_RE.sub("", first).strip()
        name_compact = recipe_name.replace(" ", "")
        first_compact = first_stripped.replace(" ", "")
        # 첫 줄이 데이터 라인(콤마/숫자 포함)이면 보존. 거의 요리명만 있을 때만 제거.
        is_data_line = ("," in first_stripped) or bool(re.search(r"\d", first_stripped))
        if (not is_data_line and first_compact and
            (first_compact == name_compact or
             (len(first_compact) >= 2 and
              (first_compact in name_compact or name_compact in first_compact)))):
            lines = lines[1:]
        text = "\n".join(lines)

    text = BULLET_RE.sub("", text)
    # 라인 앞 [N인분] 같은 분량 prefix 제거 (헤더로 의미 없음)
    text = "\n".join(LINE_PREFIX_PORTION_RE.sub("", ln) for ln in text.split("\n"))
    text = MULTI_NL_RE.sub("\n", text)
    text = "\n".join(WS_RE.sub(" ", ln).strip() for ln in text.split("\n"))
    text = text.strip()
    return text


def is_missing(raw: str | None) -> bool:
    if raw is None:
        return True
    s = raw.strip()
    if len(s) <= 1:
        return True
    if s in (".", "-", "_"):
        return True
    return False


def classify_header(header: str) -> str | None:
    """헤더 텍스트 → 섹션 키. 매칭 안 되면 None."""
    h = header.strip().strip("[]()").strip()
    if not h:
        return None
    for kw in GARNISH_KEYWORDS:
        if kw in h:
            return "garnish"
    for kw in SAUCE_KEYWORDS:
        if kw in h:
            return "sauce"
    if PORTION_HEADERS_RE.match(h):
        return "main"
    return None


def split_into_section_chunks(text: str, rcp_seq: str, report: dict, flags: dict):
    """라인 단위로 섹션을 나눠 (section, chunk_text) 리스트 반환."""
    chunks: list[tuple[str, str]] = []
    current_section = "main"
    current_buf: list[str] = []

    def flush():
        if current_buf:
            chunks.append((current_section, " , ".join(current_buf)))
            current_buf.clear()

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        # "양념장 : ..." 같은 인라인 섹션
        m = SECTION_INLINE_RE.match(line)
        if m:
            head, body = m.group(1).strip(), m.group(2).strip()
            sec = classify_header(head)
            if sec is None:
                # 알 수 없는 헤더지만 콜론으로 명시됨 → main 으로 합치고 마킹
                # 단, 머리부분이 너무 짧지 않고 재료 같지 않을 때만 헤더로 본다.
                if len(head) <= 12 and not re.search(r"\d", head):
                    flags["had_unknown_header"] = True
                    report["unknown_headers"].append(
                        {"rcp_seq": rcp_seq, "header": head}
                    )
                    flush()
                    current_section = "main"
                    current_buf.append(body)
                    continue
                # 헤더로 보기 애매하면 라인 전체를 항목으로
                current_buf.append(line)
                continue
            flush()
            current_section = sec
            current_buf.append(body)
            continue

        # 단독 헤더 라인 (콜론 없이)
        hm = HEADER_RE.match(line)
        if hm and "," not in line and not re.search(r"\d", line) and len(line) <= 14:
            head = hm.group(1)
            sec = classify_header(head)
            if sec is not None:
                flush()
                current_section = sec
                continue
            # 알 수 없는 헤더 단독 라인
            if not any(kw in head for kw in ("재료",)) and len(head) <= 10:
                flags["had_unknown_header"] = True
                report["unknown_headers"].append(
                    {"rcp_seq": rcp_seq, "header": head}
                )
                flush()
                current_section = "main"
                continue

        current_buf.append(line)

    flush()
    return chunks


def split_items(chunk: str) -> list[str]:
    """콤마로 항목 분리. 괄호 안 콤마는 보존."""
    items: list[str] = []
    buf = []
    depth = 0
    for ch in chunk:
        if ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch == "," and depth == 0:
            s = "".join(buf).strip()
            if s:
                items.append(s)
            buf = []
        else:
            buf.append(ch)
    s = "".join(buf).strip()
    if s:
        items.append(s)
    return items


def find_last_paren(text: str) -> tuple[str, str] | None:
    """마지막 짝 맞는 괄호의 (앞부분, 괄호내용) 반환. 없으면 None."""
    if not text.endswith(")"):
        return None
    depth = 0
    for i in range(len(text) - 1, -1, -1):
        ch = text[i]
        if ch == ")":
            depth += 1
        elif ch == "(":
            depth -= 1
            if depth == 0:
                return text[:i].rstrip(), text[i + 1:-1].strip()
    return None


AMOUNT_TAIL_RE = re.compile(
    r"""
    (?P<amount>
        (?:약\s*)?
        (?:\d+(?:[./]\d+)?(?:\s*\d+/\d+)?)
        \s*
        (?:[a-zA-Z%]+|g|kg|ml|L|cc|컵|큰술|작은술|T|t|Ts|ts|개|장|모|쪽|줄기|줌|마리|봉|대|단|꼬집|숟갈|티스푼|테이블스푼|알|입|덩이|덩어리|토막|쪽|뿌리|cm|mm)?
    )$
    """,
    re.VERBOSE,
)

NUMERIC_AMOUNT_RE = re.compile(r"\d")


def parse_item(text: str) -> dict | None:
    """단일 항목 텍스트 → {name, amount, note} dict."""
    text = text.strip().strip(",").strip()
    if not text:
        return None

    note: str | None = None
    amount: str | None = None
    name: str

    paren = find_last_paren(text)
    if paren is not None:
        head, inner = paren
        # 괄호 안에 숫자 또는 비수치 수량 표현이 있으면 amount 후보
        inner_has_qty = (
            NUMERIC_AMOUNT_RE.search(inner) is not None
            or any(kw in inner for kw in NON_NUMERIC_AMOUNTS)
        )
        if inner_has_qty:
            # head 끝에 수량 표현이 또 있는지 확인 (수량 뒤 괄호 패턴)
            tail_m = AMOUNT_TAIL_RE.search(head)
            if tail_m and tail_m.group("amount").strip():
                amount = tail_m.group("amount").strip()
                name = head[:tail_m.start()].strip()
                note = inner if inner else None
            else:
                amount = inner
                name = head
        else:
            # 괄호 안이 수량 아님 → 이름의 일부로 본다
            name = text
    else:
        name = text

    if amount is None and name:
        # 괄호 없는 형태: 비수치 수량 토큰이 끝에 있는지
        for kw in NON_NUMERIC_AMOUNTS:
            if name.endswith(kw):
                amount = kw
                name = name[:-len(kw)].strip()
                break
        else:
            tail_m = AMOUNT_TAIL_RE.search(name)
            if tail_m:
                cand = tail_m.group("amount").strip()
                # 단순 숫자만 있으면 단위가 없는 거라 amount로 의심스럽지만 일단 보존
                rest = name[:tail_m.start()].strip()
                if rest:
                    amount = cand
                    name = rest

    name = name.strip().strip(",").strip()
    if not name and amount:
        # name이 비고 amount만 남으면 비정상 → name에 통째로
        name = amount
        amount = None

    if not name:
        return None

    return {"name": name, "amount": amount, "note": note}


def parse_one(recipe: dict, report: dict) -> dict:
    """단일 레시피 파싱. enrichment_flags 등 추가 필드 dict 반환."""
    rcp_seq = str(recipe.get("rcp_seq", ""))
    name = recipe.get("name", "") or ""
    raw = recipe.get("ingredients", "")

    flags = {
        "ingredients_missing": False,
        "ingredients_parse_failed": False,
        "had_html": False,
        "had_unicode_fraction": False,
        "had_unknown_header": False,
    }
    structured = {"main": [], "sauce": [], "garnish": []}

    if is_missing(raw):
        flags["ingredients_missing"] = True
        return {
            "ingredients_structured": structured,
            "ingredients_clean": "",
            "ingredient_count_total": 0,
            "ingredient_count_main": 0,
            "enrichment_flags": flags,
        }

    text = normalize_text(raw, name, flags)
    if not text:
        flags["ingredients_parse_failed"] = True
        return {
            "ingredients_structured": structured,
            "ingredients_clean": "",
            "ingredient_count_total": 0,
            "ingredient_count_main": 0,
            "enrichment_flags": flags,
        }

    chunks = split_into_section_chunks(text, rcp_seq, report, flags)
    for section, chunk in chunks:
        for raw_item in split_items(chunk):
            parsed = parse_item(raw_item)
            if parsed is not None:
                structured[section].append(parsed)

    total = sum(len(v) for v in structured.values())
    if total == 0:
        flags["ingredients_parse_failed"] = True

    clean_parts: list[str] = []
    for sec in ("main", "sauce", "garnish"):
        for it in structured[sec]:
            if it["amount"]:
                clean_parts.append(f"{it['name']} {it['amount']}")
            else:
                clean_parts.append(it["name"])
    clean_str = ", ".join(clean_parts)

    return {
        "ingredients_structured": structured,
        "ingredients_clean": clean_str,
        "ingredient_count_total": total,
        "ingredient_count_main": len(structured["main"]),
        "enrichment_flags": flags,
    }


def add_sample(report: dict, key: str, rcp_seq: str, limit: int = 5):
    bucket = report["samples_by_pattern"].setdefault(key, [])
    if len(bucket) < limit and rcp_seq not in bucket:
        bucket.append(rcp_seq)


def run(sample: int | None):
    if not SRC_PATH.exists():
        raise SystemExit(f"입력 파일 없음: {SRC_PATH}")

    with SRC_PATH.open("r", encoding="utf-8") as f:
        recipes = json.load(f)

    if sample is not None:
        recipes_to_process = recipes[:sample]
    else:
        recipes_to_process = recipes

    report = {
        "total": len(recipes_to_process),
        "missing": 0,
        "parse_failed": 0,
        "section_distribution": {
            "main_only": 0,
            "with_sauce": 0,
            "with_garnish": 0,
        },
        "unknown_headers": [],
        "non_numeric_amounts": 0,
        "had_html_count": 0,
        "had_unicode_fraction_count": 0,
        "samples_by_pattern": OrderedDict([
            ("html_tag", []),
            ("unicode_fraction", []),
            ("section_sauce", []),
            ("section_garnish", []),
            ("non_numeric_amount", []),
            ("missing", []),
            ("unknown_header", []),
            ("parse_failed", []),
        ]),
    }

    out_recipes = []
    for recipe in recipes_to_process:
        rcp_seq = str(recipe.get("rcp_seq", ""))
        extra = parse_one(recipe, report)
        merged = {**recipe, **extra}
        out_recipes.append(merged)

        flags = extra["enrichment_flags"]
        structured = extra["ingredients_structured"]

        if flags["ingredients_missing"]:
            report["missing"] += 1
            add_sample(report, "missing", rcp_seq)
        if flags["ingredients_parse_failed"]:
            report["parse_failed"] += 1
            add_sample(report, "parse_failed", rcp_seq)
        if flags["had_html"]:
            report["had_html_count"] += 1
            add_sample(report, "html_tag", rcp_seq)
        if flags["had_unicode_fraction"]:
            report["had_unicode_fraction_count"] += 1
            add_sample(report, "unicode_fraction", rcp_seq)
        if flags["had_unknown_header"]:
            add_sample(report, "unknown_header", rcp_seq)

        has_sauce = len(structured["sauce"]) > 0
        has_garnish = len(structured["garnish"]) > 0
        has_main = len(structured["main"]) > 0
        if has_main and not has_sauce and not has_garnish:
            report["section_distribution"]["main_only"] += 1
        if has_sauce:
            report["section_distribution"]["with_sauce"] += 1
            add_sample(report, "section_sauce", rcp_seq)
        if has_garnish:
            report["section_distribution"]["with_garnish"] += 1
            add_sample(report, "section_garnish", rcp_seq)

        for sec_items in structured.values():
            for it in sec_items:
                amt = it.get("amount")
                if amt and not NUMERIC_AMOUNT_RE.search(amt):
                    report["non_numeric_amounts"] += 1
                    add_sample(report, "non_numeric_amount", rcp_seq)
                    break

    suffix = "_sample" if sample is not None else ""
    out_recipes_path = DATA_DIR / f"recipes_cleaned{suffix}.json"
    out_report_path = DATA_DIR / f"parse_report{suffix}.json"

    with out_recipes_path.open("w", encoding="utf-8") as f:
        json.dump(out_recipes, f, ensure_ascii=False, indent=2)
    with out_report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"처리 완료: {report['total']}개")
    print(f"  missing       : {report['missing']}")
    print(f"  parse_failed  : {report['parse_failed']}")
    print(f"  section_dist  : {report['section_distribution']}")
    print(f"출력: {out_recipes_path.name}, {out_report_path.name}")


def main():
    p = argparse.ArgumentParser(description="recipes.json ingredients 파서")
    p.add_argument("--sample", type=int, default=None, help="앞 N개만 처리")
    args = p.parse_args()
    run(args.sample)


if __name__ == "__main__":
    main()
