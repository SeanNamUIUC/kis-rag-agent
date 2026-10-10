"""
질문으로 비슷한 청크 찾기 (벡터 검색)

흐름:
    질문 --bge-m3--> 질문 벡터
      -> Chroma에서 가장 가까운 청크 k개 (필터가 있으면 그 범위 안에서만)
      -> [{chunk_id, score, text, name, report_nm, section_path, ...}, ...]

질문에는 헤더를 붙이지 않는다. 헤더는 청크 쪽에만 붙여서
"이 청크가 어느 회사·보고서 것인지"를 좌표에 담는 용도다.

사용법 (프로젝트 루트에서):
    python -m kisrag.retrieve.vector "원재료 가격 상승 영향"
    python -m kisrag.retrieve.vector "원재료 가격 상승 영향" --ticker 005930
    python -m kisrag.retrieve.vector "최근 자기주식 취득" --ticker 071050 --since 20260101
"""

import argparse
import time
from functools import lru_cache

import chromadb
import numpy as np

from kisrag.index.embed import CHROMA_DIR, COLLECTION, embed_texts


@lru_cache(maxsize=1)
def get_collection():
    # Chroma 컬렉션을 한 번만 열고 재사용 (속도 향상)
    return chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLLECTION)


def build_where(ticker: str | None = None, since: int | None = None, doc_type: str | None = None):
    """필터 조건 만들기. Chroma는 조건이 2개 이상일 때만 $and로 묶어야 한다."""
    conds = []
    if ticker:
        conds.append({"ticker": ticker})
    if since:
        conds.append({"rcept_dt": {"$gte": int(since)}})
    if doc_type:
        conds.append({"pblntf_ty": doc_type})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


def search(query: str, k: int = 5, ticker: str | None = None,
           since: int | None = None, doc_type: str | None = None) -> list[dict]:
    q = embed_texts([query])[0]
    res = get_collection().query(
        query_embeddings=[q.tolist()],
        n_results=k,
        where=build_where(ticker, since, doc_type),
        include=["documents", "metadatas", "embeddings"],# 청크 본문, 메타데이터, 벡터까지 다 가져오기
    )
    hits = []
    for cid, doc, meta, emb in zip(res["ids"][0], res["documents"][0],
                                   res["metadatas"][0], res["embeddings"][0]):
        # 점수는 직접 계산: 둘 다 길이 1이라 내적 = 코사인 유사도 (실험에서 본 그 점수)
        score = float(np.dot(q, np.asarray(emb, dtype=np.float32)))
        hits.append({"chunk_id": cid, "score": score, "text": doc, **meta})
    return hits


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--ticker", help="종목코드로 범위 제한 (예: 005930)")
    ap.add_argument("--since", type=int, help="이 날짜 이후 공시만 (예: 20260101)")
    args = ap.parse_args()

    search("워밍업")  # 첫 호출은 모델 로딩 때문에 느리니 한 번 미리 돌림
    t0 = time.time()
    hits = search(args.query, k=args.k, ticker=args.ticker, since=args.since)
    print(f"검색 {(time.time() - t0) * 1000:.0f} ms\n")
    for h in hits:
        print(f"[{h['chunk_id']}] {h['score']:.3f} | {h['name']} | {h['report_nm']} | {h['section_path']}")
        print("    " + h["text"][:150].replace("\n", " / "))
        print()