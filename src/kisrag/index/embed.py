"""
청크를 벡터로 바꿔 Chroma(벡터DB)에 저장

흐름:
    data/processed/chunks.jsonl
      -> 각 청크의 embed_text(헤더 + 본문)를 bge-m3로 1,024차원 벡터로 변환 (GPU)
      -> Chroma에 (chunk_id, 벡터, 본문 text, 메타데이터) 한 묶음씩 저장
      -> data/chroma/ 폴더에 디스크로 저장됨 (다시 켜도 그대로)

왜 embed_text로 벡터를 만들고 document에는 text를 넣나:
    - 벡터(찾기용): 헤더가 있어야 어느 회사·보고서 청크인지가 좌표에 담김
    - document(읽기용): 검색 결과로 꺼내서 LLM에게 보여줄 깨끗한 본문

사용법 (프로젝트 루트에서):
    python -m kisrag.index.embed            # 전체 (처음 한 번, 몇 분)
    python -m kisrag.index.embed --limit 2000   # 빠른 테스트용 일부만
"""

import argparse
import json
import time
from functools import lru_cache
from pathlib import Path

import chromadb
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

CHUNKS = Path("data/processed/chunks.jsonl")
CHROMA_DIR = Path("data/chroma")
COLLECTION = "dart_chunks"
MODEL_NAME = "BAAI/bge-m3"
BATCH = 64          # GPU에 한 번에 넣는 청크 수
ADD_BATCH = 1000    # Chroma에 한 번에 넣는 개수


@lru_cache(maxsize=1)  # 모델은 한 번만 불러오고 재사용 (2GB라 매번 부르면 느림)
def get_embedder() -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    if device == "cuda":
        model.half()            # 16비트: 메모리 절반, 속도 향상
    model.max_seq_length = 1024  # 청크 최대 1,500자 + 헤더 ≈ 1,000토큰 이하
    return model


def embed_texts(texts: list[str], show_progress: bool = False) -> np.ndarray:
    """글자 리스트 -> (개수, 1024) 벡터. 길이를 1로 맞춰서 내적 = 코사인 유사도."""
    vecs = get_embedder().encode(
        texts,
        batch_size=BATCH,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
    )
    return vecs.astype(np.float32)  # Chroma는 32비트 실수를 받음


def to_metadata(c: dict) -> dict:
    """Chroma 메타데이터는 문자열·숫자·참거짓만 가능 (None 불가)."""
    return {
        "ticker": c["ticker"],
        "name": c["name"],
        "rcept_no": c["rcept_no"],
        "rcept_dt": int(c["rcept_dt"]),   # 숫자로 저장해야 "이 날짜 이후" 같은 범위 필터가 됨
        "report_nm": c["report_nm"],
        "pblntf_ty": c["pblntf_ty"],
        "section_path": c["section_path"],
        "is_table": bool(c["is_table"]),
        "unit": c.get("unit") or "",
        "chars": int(c["chars"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="앞에서 N개만 (테스트용)")
    args = ap.parse_args()

    chunks = [json.loads(l) for l in CHUNKS.open(encoding="utf-8")]
    if args.limit:
        chunks = chunks[: args.limit]
    print(f"청크 {len(chunks):,}개 읽음")

    # 1) 벡터 만들기 (GPU)
    t0 = time.time()
    get_embedder()
    print(f"모델 로딩 {time.time() - t0:.1f}초")

    t0 = time.time()
    vecs = embed_texts([c["embed_text"] for c in chunks], show_progress=True)
    dt = time.time() - t0
    print(f"임베딩 {dt:.1f}초 | 초당 {len(chunks) / dt:,.0f}개 | 벡터 {vecs.shape}")
    if torch.cuda.is_available():
        print(f"GPU 최대 메모리 {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # 2) Chroma에 저장. 다시 돌리면 기존 컬렉션을 지우고 새로 만든다 (청크가 바뀌었을 수 있으니)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass  # 처음이면 지울 게 없음
    col = client.create_collection(COLLECTION)

    t0 = time.time()
    for i in range(0, len(chunks), ADD_BATCH):
        part = chunks[i : i + ADD_BATCH]
        col.add(
            ids=[c["chunk_id"] for c in part],
            embeddings=vecs[i : i + ADD_BATCH].tolist(),#1000 * 1024
            documents=[c["text"] for c in part],        # LLM이 읽을 본문
            metadatas=[to_metadata(c) for c in part],
        )
    print(f"Chroma 저장 {time.time() - t0:.1f}초 | 저장된 개수 {col.count():,} | 위치 {CHROMA_DIR}/")


if __name__ == "__main__":
    main()