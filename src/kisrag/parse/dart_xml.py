"""
DART 공시 원문 XML 파서

원문 구조 (dart4.xsd):
    DOCUMENT
      DOCUMENT-NAME, COMPANY-NAME
      BODY
        COVER                     <- 표지 (건너뜀)
        SECTION-1                 <- 대목차 (I. 회사의 개요 ...)
          TITLE
          P / TABLE / TABLE-GROUP / LIBRARY
          SECTION-2               <- 중목차
            TITLE ...

결과: 블록 리스트. 블록 하나 = 문단 하나 또는 표 하나.
    {"path": ["II. 사업의 내용", "3. 원재료 및 생산설비"],
     "type": "text" | "table",
     "text": "...",            # 표는 마크다운 표 문자열
     "unit": "백만원" | None}   # 표의 단위 (있을 때만)
"""

import re
from pathlib import Path

from lxml import etree

from kisrag.parse.normalize import clean_ws, find_unit, normalize_number

CELL_TAGS = {"TH", "TD", "TE", "TU"}   # 표 칸 종류 4가지
CONTAINER_TAGS = {"TABLE-GROUP", "LIBRARY", "SPAN"}  # 안을 들여다볼 그릇
SKIP_TITLES = ("목 차", "목차", "대표이사 등의 확인")  # 검색 가치 없는 섹션
MAX_REPEAT_CHARS = 30  # 병합 칸에 복사할 최대 글자 수

# 원문 본문의 "<산업의 성장성>", "<MOU 체결>", "<1>" 같은 꺾쇠 글자를 파서가 태그로 착각하면
# 영원히 안 닫히는 태그가 되고, recover 모드가 문서 나머지 전체를 그 안에 넣어버린다.
# 판단 기준: 그 파일 안에 닫는 태그(</이름>)나 스스로 닫는 태그(<이름 .../>)로
# 실제로 등장하는 이름만 진짜 태그로 인정한다. </MOU>는 어디에도 없으니 <MOU 체결>은 글자.
_CLOSE_TAG = re.compile(rb"</([A-Za-z][\w.-]*)\s*>")
_SELF_TAG = re.compile(rb"<([A-Za-z][\w.-]*)\b[^<>]*/>")


def _stray_lt_pattern(raw: bytes):
    names = set(_CLOSE_TAG.findall(raw)) | set(_SELF_TAG.findall(raw))
    if not names:
        return re.compile(rb"<(?![!?])")
    alt = b"|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    # '<' 다음이 (/)진짜태그이름 + (공백 | / | >) 이거나, <!-- 주석 / <?xml 선언이면 진짜. 나머지는 글자.
    return re.compile(rb"<(?!(?:/?(?:" + alt + rb")[\s/>])|[!?])")

# 표기의 시작이 아닌 '&' : "R&D", "M&A" 같은 글자. 그대로 두면 파서가 그 부분을 버린다.
# 진짜 표기는 &amp; &lt; &#39; &#x27; 처럼 '&이름;' 또는 '&#숫자;' 모양이다.
_STRAY_AMP = re.compile(rb"&(?!(?:[A-Za-z][A-Za-z0-9]*|#[0-9]+|#x[0-9A-Fa-f]+);)")


def _tag(el) -> str:
    """주석·처리명령 같은 비(非)태그 노드는 빈 문자열."""
    return el.tag if isinstance(el.tag, str) else ""


def _text(el) -> str:
    return clean_ws("".join(el.itertext()))


def _to_int(v, default=1) -> int:
    try:
        return max(1, int(v))
    except (TypeError, ValueError):
        return default


def table_to_rows(table) -> list[list[str]]:
    """TABLE -> 2차원 리스트. 병합 칸(COLSPAN/ROWSPAN)은 같은 값으로 채워서 행·열 관계를 유지."""
    rows, pending = [], {}  # pending[열번호] = (남은 행 수, 값)  <- ROWSPAN 이월분
    for tr in table.iter("TR"):
        cells = [c for c in tr if _tag(c) in CELL_TAGS]
        row, col, i = [], 0, 0
        while i < len(cells) or col in pending:
            if col in pending:                      # 위 행에서 내려온 병합 칸
                left, val = pending.pop(col)
                if left > 1:
                    pending[col] = (left - 1, val)
                row.append(val)
                col += 1
                continue
            c = cells[i]
            i += 1
            val = normalize_number(_text(c))
            colspan, rowspan = _to_int(c.get("COLSPAN")), _to_int(c.get("ROWSPAN"))
            for k in range(colspan):
                # 짧은 머리글("금액")만 병합된 칸마다 복사하고, 긴 글은 첫 칸에만 둔다.
                # 긴 글을 복사하면 같은 문장이 수십 번 반복돼 청크가 폭발함.
                v = val if (k == 0 or len(val) <= MAX_REPEAT_CHARS) else ""
                if rowspan > 1:
                    pending[col] = (rowspan - 1, v)
                row.append(v)
                col += 1
        if any(v for v in row):  # 완전히 빈 행은 버림
            rows.append(row)
    return rows


def rows_to_markdown(rows: list[list[str]]) -> str:
    width = max(len(r) for r in rows)
    lines = []
    for k, r in enumerate(rows):
        r = [v.replace("|", "/") for v in r] + [""] * (width - len(r))
        lines.append("| " + " | ".join(r) + " |")
        if k == 0:
            lines.append("|" + "---|" * width)
    return "\n".join(lines)


class _SectionWalker:
    """섹션 하나를 순서대로 훑으며 블록을 만든다.

    - 섹션의 첫 TITLE  -> 이 섹션의 이름 (path에 추가)
    - 그다음 TITLE들   -> 소제목 (예: 주석 '8. 재고자산'), 이전 소제목을 대체
    """

    def __init__(self, parent_path: list[str], out: list[dict]):
        self.base = list(parent_path)
        self.cur = list(parent_path)
        self.has_title = False
        self.pending_unit = None
        self.out = out

    def walk(self, node):
        for child in node:
            tag = _tag(child)
            if not tag:
                continue
            if tag.startswith("SECTION-"):
                _SectionWalker(self.base, self.out).walk(child)
                self.cur = list(self.base)
            elif tag == "TITLE":
                title = _text(child)
                if not title:
                    continue
                if not self.has_title:          # 첫 TITLE = 이 섹션 이름
                    self.base = self.base + [title]
                    self.cur = list(self.base)
                    self.has_title = True
                else:                           # 다음 TITLE들 = 소제목
                    self.cur = self.base + [title]
            elif tag == "P":
                self._paragraph(child)
            elif tag == "TABLE":
                self._table(child)
            elif tag in CONTAINER_TAGS:
                self.walk(child)  # 그릇은 같은 섹션으로 취급하고 안을 계속 훑음

    def _skip(self) -> bool:
        return any(s in " ".join(self.cur) for s in SKIP_TITLES)

    def _emit(self, typ: str, text: str, unit=None):
        if text and not self._skip():
            self.out.append({"path": list(self.cur), "type": typ, "text": text, "unit": unit})

    def _paragraph(self, p):
        text = _text(p)
        unit = find_unit(text)
        if unit:                      # "(단위 : 백만원)" 한 줄짜리 문단 -> 다음 표에 붙임
            self.pending_unit = unit
            return
        self._emit("text", text)

    def _table(self, table):
        rows = table_to_rows(table)
        if not rows:
            return
        flat = " ".join(v for r in rows for v in r)
        unit = find_unit(flat)
        if unit and len(flat) <= 40:  # 단위만 적힌 작은 표 -> 다음 표에 붙임
            self.pending_unit = unit
            return
        unit = unit or self.pending_unit
        self.pending_unit = None
        md = rows_to_markdown(rows)
        if unit:
            md = f"(단위: {unit})\n{md}"
        self._emit("table", md, unit)


def _misplaced_sub(sec) -> bool:
    """중목차의 부모를 거슬러 올라가 LIBRARY를 건너뛰었을 때 상위 섹션이 아니면 잘못 놓인 것."""
    p = sec.getparent()
    while p is not None and p.tag == "LIBRARY":
        p = p.getparent()
    return p is None or not str(p.tag).startswith("SECTION-")


def load_tree(xml_path: Path):
    """원문을 읽어 XML 트리로. 깨진 구조를 두 단계로 복구한다.

    1) 원인 제거: 태그가 아닌 '<'·'&'를 글자(&lt; &amp;)로 바꾼 뒤 파싱
    2) 안전망: 그래도 엉뚱한 곳(표 칸, 문단 등)에 끼어 들어간 대목차(SECTION-1)는
       BODY 바로 아래로 꺼내고, 대목차 없이 BODY에 떨어진 중목차(SECTION-2 등)는
       문서 순서상 바로 앞 대목차 안으로 넣는다

    돌려주는 값: (root, {"escaped_lt": 바꾼 '<' 개수, "escaped_amp": 바꾼 '&' 개수,
                         "moved_sections": 꺼낸 대목차 수})
    """
    raw = Path(xml_path).read_bytes()
    raw = raw.replace(b"&nbsp;", b" ")                    # HTML 띄어쓰기 표기 (XML엔 없음)
    raw, n_escaped = _stray_lt_pattern(raw).subn(b"&lt;", raw)
    raw, n_amp = _STRAY_AMP.subn(b"&amp;", raw)
    parser = etree.XMLParser(recover=True, huge_tree=True, encoding="utf-8")
    root = etree.fromstring(raw, parser)

    moved = 0
    body = root.find("BODY")
    if body is not None:
        # 옮기기 전에 문서 순서를 기록 (옮기는 동안 트리 모양이 바뀌므로)
        order = {el: i for i, el in enumerate(root.iter())
                 if isinstance(el.tag, str) and el.tag.startswith("SECTION-")}
        lost_s1 = [s for s in root.iter("SECTION-1") if s.getparent() is not body]
        # 중목차(SECTION-2 이하)는 원래 상위 섹션이나 LIBRARY 안에만 있어야 한다.
        # BODY에 떨어졌거나, 표 칸·문단 안으로 빨려 들어간 것은 잘못 놓인 것.
        orphans = [s for s in order if s.tag != "SECTION-1" and _misplaced_sub(s)]
        if lost_s1 or orphans:
            units = sorted(list(root.iter("SECTION-1")) + orphans, key=order.get)
            last_s1 = None
            for u in units:                      # 문서 순서대로 다시 세움
                if u.tag == "SECTION-1":
                    body.append(u)               # lxml의 append는 '이동' (원래 자리에서 빠짐)
                    last_s1 = u
                else:
                    (last_s1 if last_s1 is not None else body).append(u)
            moved = len(lost_s1) + len(orphans)
    return root, {"escaped_lt": n_escaped, "escaped_amp": n_amp, "moved_sections": moved}


def parse_document(xml_path: Path) -> dict:
    """공시 원문 XML 하나 -> {"doc_name", "company", "blocks", "repairs"}."""
    root, repairs = load_tree(xml_path)

    doc_name = clean_ws(root.findtext("DOCUMENT-NAME") or "")
    company = clean_ws(root.findtext("COMPANY-NAME") or "")
    body = root.find("BODY")
    blocks: list[dict] = []
    if body is None:
        return {"doc_name": doc_name, "company": company, "blocks": blocks, "repairs": repairs}

    loose = _SectionWalker([doc_name or "본문"], blocks)  # SECTION 밖에 바로 놓인 내용용
    for child in body:
        tag = _tag(child)
        if tag == "COVER" or not tag:
            continue
        if tag.startswith("SECTION-"):
            _SectionWalker([], blocks).walk(child)
        else:
            loose.walk([child])
    return {"doc_name": doc_name, "company": company, "blocks": blocks, "repairs": repairs}


if __name__ == "__main__":
    # 사용법: python -m kisrag.parse.dart_xml data/raw/005930/접수번호/접수번호.xml
    import sys
    from collections import Counter

    doc = parse_document(Path(sys.argv[1]))
    print(doc["doc_name"], doc["company"], f"블록 {len(doc['blocks'])}개", "| 복구:", doc["repairs"])
    print(Counter(b["type"] for b in doc["blocks"]))
    for b in doc["blocks"][:15]:
        print(" > ".join(b["path"]), "|", b["type"], "|", b["text"][:60].replace("\n", " "))