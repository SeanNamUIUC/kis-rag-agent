"""
금융 문서 텍스트·숫자 정규화

- 공백 정리: "목     차" -> "목 차"
- 회계식 음수 표기: "(1,234)" -> "-1,234", "△1,234" -> "-1,234"
- 단위 문구 찾기: "(단위 : 백만원)" -> "백만원"
"""

import re

_WS = re.compile(r"\s+")
_PAREN_NEG = re.compile(r"^\(\s*([\d,]+(?:\.\d+)?)\s*\)$")      # (1,234)
_TRI_NEG = re.compile(r"^△\s*([\d,]+(?:\.\d+)?)$")               # △1,234
_UNIT = re.compile(r"단위\s*[:：]?\s*([^)\]\n]+?)\s*[)\]]?\s*$")  # (단위 : 백만원)


def clean_ws(text: str) -> str:
    """연속 공백·줄바꿈을 공백 하나로."""
    return _WS.sub(" ", text or "").strip()


def normalize_number(cell: str) -> str:
    """표 칸이 '숫자 하나'일 때만 회계식 음수를 마이너스로 바꾼다. 나머지는 그대로."""
    t = cell.strip()
    m = _PAREN_NEG.match(t) or _TRI_NEG.match(t)
    return f"-{m.group(1)}" if m else cell


def find_unit(text: str):
    """'(단위 : 백만원)' 같은 짧은 문구면 단위('백만원')를 돌려주고, 아니면 None."""
    t = clean_ws(text)
    if not t or len(t) > 40 or "단위" not in t:
        return None
    m = _UNIT.search(t)
    return m.group(1).strip() if m else None