"""
공시 원문 -> 검색용 청크

흐름:
    data/meta/{ticker}.jsonl 에서 정기공시(A)·주요사항보고(B) 목록을 읽고
    -> data/raw/{ticker}/{rcept_no}/{rcept_no}.xml 을 파싱 (첨부 _00760 등은 제외)
    -> 같은 목차 경로의 문단끼리 묶어 길이 기준으로 자르고, 표는 표 단위로 자름
    -> 청크마다 "[회사 | 보고서 | 목차경로]" 헤더를 붙여 data/processed/chunks.jsonl 저장

길이는 일단 글자 수 기준 (한국어는 bge-m3 기준 대략 1.5~2글자 ≈ 1토큰).
6일차에 크기별 Recall@5를 비교해서 확정한다.

사용법 (프로젝트 루트에서):
    python -m kisrag.index.chunker
    python -m kisrag.index.chunker --tickers 005930 071050   # 일부 종목만
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from kisrag.parse.dart_xml import parse_document

DATA = Path("data")
OUT = DATA / "processed" / "chunks.jsonl"

TARGET_CHARS = 1000   # 이 정도 차면 청크를 끊음
MAX_CHARS = 1500      # 한 청크의 최대 길이
OVERLAP_CHARS = 150   # 앞 청크 끝부분을 다음 청크 앞에 겹쳐 넣는 양
MIN_CHARS = 10        # 이보다 짧은 청크는 버림 ("해당사항 없습니다."(10자)는 남고 "해당사항 없음"(7자)은 버려짐)
DOC_TYPES = {"A", "B"}


# ---------- 자르기 ----------

def split_text(paras: list[str]) -> list[str]:
    """문단 리스트를 TARGET_CHARS 근처에서 끊는다. 끊을 때 마지막 문단 일부를 겹쳐서 문맥 유지."""
    chunks, cur, cur_len = [], [], 0
    for p in paras:
        if len(p) > MAX_CHARS:                       # 문단 하나가 너무 길면 그 자체를 잘라냄
            if cur:
                chunks.append("\n".join(cur))
                cur, cur_len = [], 0
            step = MAX_CHARS - OVERLAP_CHARS
            chunks += [p[i:i + MAX_CHARS] for i in range(0, len(p), step)]
            continue
        if cur and cur_len + len(p) > TARGET_CHARS:
            chunks.append("\n".join(cur))
            tail, tail_len = [], 0                   # 겹칠 꼬리 문단들
            for q in reversed(cur):
                if tail_len + len(q) > OVERLAP_CHARS:
                    break
                tail.insert(0, q)
                tail_len += len(q)
            cur, cur_len = tail, tail_len
        if cur and cur_len + len(p) + len(cur) > MAX_CHARS:
            # 겹침 꼬리 + 이번 문단이 최대 길이를 넘으면 겹침을 포기 (len(cur)는 줄바꿈 몫)
            cur, cur_len = [], 0
        cur.append(p)
        cur_len += len(p)
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def split_table(md: str) -> list[str]:
    """큰 표는 행 묶음으로 자르고, 조각마다 머리글(단위·헤더 행)을 반복해서 붙인다."""
    if len(md) <= MAX_CHARS:
        return [md]
    lines = md.split("\n")
    n_head = 3 if lines[0].startswith("(단위") else 2   # (단위) + 헤더행 + 구분선
    head, body = lines[:n_head], lines[n_head:]
    if not body:                        # 행이 하나뿐인 표 (칸 하나에 긴 글) -> 글자 기준 분할
        return split_text(lines)
    parts, cur = [], []
    head_len = sum(len(h) + 1 for h in head)
    for line in body:
        if cur and head_len + sum(len(c) + 1 for c in cur) + len(line) > MAX_CHARS:
            parts.append("\n".join(head + cur))
            cur = []
        cur.append(line)
    if cur:
        parts.append("\n".join(head + cur))

    # 그래도 넘치는 조각(행 하나가 아주 길거나 머리글이 거대한 경우)은 글자 기준으로 강제 분할
    out = []
    for p in parts:
        out += [p] if len(p) <= MAX_CHARS else split_text(p.split("\n"))
    return out


def keep_block(b: dict) -> bool:
    """RAG에 넣을 블록만 남긴다.

    - III. 재무에 관한 사항의 표: 숫자는 재무 DB(정형 데이터)로 답할 것이므로 제외.
      단, '요약재무정보' 표는 핵심 수치 요약이라 남김. 주석의 '설명 문단'도 남김.
    - XII. 상세표: 계열회사·출자 현황 등 거대한 목록 표라 제외.
    """
    path = " > ".join(b["path"])
    if "상세표" in path:
        return False
    if b["type"] == "table" and "재무에 관한 사항" in path and "요약재무정보" not in path:
        return False
    return True


# ---------- 문서 하나 -> 청크들 ----------

def chunk_document(meta: dict, blocks: list[dict]) -> list[dict]:
    """같은 목차 경로에 연달아 나오는 문단은 한 덩어리로 묶어서 자르고, 표는 따로 자른다."""
    groups = []  # [(path, type, [texts], unit)]
    for b in blocks:
        last = groups[-1] if groups else None
        if b["type"] == "text" and last and last[1] == "text" and last[0] == b["path"]:
            last[2].append(b["text"])
        else:
            groups.append((b["path"], b["type"], [b["text"]], b.get("unit")))

    chunks = []
    for path, typ, texts, unit in groups:
        pieces = split_text(texts) if typ == "text" else split_table(texts[0])
        for piece in pieces:
            if len(piece) < MIN_CHARS:
                continue
            path_str = " > ".join(path)
            header = f"[{meta['name']} | {meta['report_nm']} | {path_str}]"
            chunks.append({
                "ticker": meta["ticker"],
                "name": meta["name"],
                "rcept_no": meta["rcept_no"],
                "rcept_dt": meta["rcept_dt"],
                "report_nm": meta["report_nm"],
                "pblntf_ty": meta["pblntf_ty"],
                "section_path": path_str,
                "is_table": typ == "table",
                "unit": unit,
                "text": piece,
                "embed_text": f"{header}\n{piece}",   # 임베딩·BM25에는 헤더 포함본을 씀
                "chars": len(piece),
            })
    return chunks


def main_xml(folder: Path, rcept_no: str):
    """본문 파일 = {rcept_no}.xml. 없으면 폴더에서 가장 큰 xml."""
    f = folder / f"{rcept_no}.xml"
    if f.exists():
        return f
    xmls = sorted(folder.glob("*.xml"), key=lambda p: p.stat().st_size, reverse=True)
    return xmls[0] if xmls else None


# ---------- 전체 실행 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", help="일부 종목만 (예: 005930 071050)")
    args = ap.parse_args()

    meta_files = sorted((DATA / "meta").glob("*.jsonl"))
    if args.tickers:
        meta_files = [m for m in meta_files if m.stem in args.tickers]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    stats = defaultdict(lambda: {"docs": 0, "chunks": 0, "chars": 0, "tables": 0,
                                 "dups": 0, "skipped": 0, "max": 0})
    failed = []
    repairs = {"escaped_lt": 0, "escaped_amp": 0, "moved_sections": 0, "docs_moved": 0}
    cid = 0

    with OUT.open("w", encoding="utf-8") as fout:
        for mf in meta_files:
            rows = [json.loads(l) for l in mf.open(encoding="utf-8")]
            rows = [r for r in rows if r["pblntf_ty"] in DOC_TYPES]
            rows.sort(key=lambda r: r["rcept_dt"], reverse=True)  # 최신 공시부터
            seen = set()  # 같은 종목에서 이미 나온 본문 (분기마다 반복되는 문단 제거용)

            for meta in rows:
                xml = main_xml(DATA / "raw" / meta["ticker"] / meta["rcept_no"], meta["rcept_no"])
                if xml is None:
                    failed.append((meta["name"], meta["rcept_no"], "원문 없음"))
                    continue
                try:
                    doc = parse_document(xml)
                except Exception as e:  # 깨진 파일 하나 때문에 전체가 멈추지 않게
                    failed.append((meta["name"], meta["rcept_no"], repr(e)[:80]))
                    continue

                s = stats[meta["name"]]
                s["docs"] += 1
                repairs["escaped_lt"] += doc["repairs"]["escaped_lt"]
                repairs["escaped_amp"] += doc["repairs"]["escaped_amp"]
                repairs["moved_sections"] += doc["repairs"]["moved_sections"]
                repairs["docs_moved"] += doc["repairs"]["moved_sections"] > 0

                # 중복 제거는 '문단·표 단위'로. 최신 공시에 이미 나온 (소목차, 내용) 쌍이면 버림.
                # 소목차를 같이 비교하는 이유: "해당사항 없습니다" 같은 짧은 문장이
                # 다른 섹션에서 나온 것까지 지워지지 않게.
                fresh = []
                for b in doc["blocks"]:
                    if not keep_block(b):
                        s["skipped"] += 1
                        continue
                    key = (b["path"][-1] if b["path"] else "") + "\x00" + b["text"]
                    h = hashlib.sha1(key.encode()).hexdigest()
                    if h in seen:
                        s["dups"] += 1
                        continue
                    seen.add(h)
                    fresh.append(b)

                for ch in chunk_document(meta, fresh):
                    ch["chunk_id"] = f"C{cid:06d}"
                    cid += 1
                    fout.write(json.dumps(ch, ensure_ascii=False) + "\n")
                    s["chunks"] += 1
                    s["chars"] += ch["chars"]
                    s["tables"] += ch["is_table"]
                    s["max"] = max(s["max"], ch["chars"])

    head = f"{'종목':<12}{'문서':>5}{'청크':>8}{'평균글자':>9}{'최대글자':>9}{'표비율':>8}{'중복제거':>9}{'표제외':>8}"
    print(head)
    total = {"docs": 0, "chunks": 0, "chars": 0, "tables": 0, "dups": 0, "skipped": 0, "max": 0}
    for name, s in stats.items():
        avg = s["chars"] / max(s["chunks"], 1)
        tr = s["tables"] / max(s["chunks"], 1)
        print(f"{name:<12}{s['docs']:>5}{s['chunks']:>8}{avg:>9.0f}{s['max']:>9}{tr:>8.0%}"
              f"{s['dups']:>9}{s['skipped']:>8}")
        for k in total:
            total[k] = max(total[k], s[k]) if k == "max" else total[k] + s[k]
    print(f"{'합계':<12}{total['docs']:>5}{total['chunks']:>8}"
          f"{total['chars'] / max(total['chunks'], 1):>9.0f}{total['max']:>9}"
          f"{total['tables'] / max(total['chunks'], 1):>8.0%}{total['dups']:>9}{total['skipped']:>8}")
    print(f"\n구조 복구: 태그가 아닌 '<' {repairs['escaped_lt']}곳, "
          f"표기가 아닌 '&' {repairs['escaped_amp']}곳을 글자로 바꿈, "
          f"엉뚱한 곳에 끼인 대목차 {repairs['moved_sections']}개를 제자리로 ({repairs['docs_moved']}개 문서)")
    print(f"저장: {OUT}")
    if failed:
        print(f"\n실패 {len(failed)}건:")
        for f in failed[:10]:
            print("  ", *f)


if __name__ == "__main__":
    main()