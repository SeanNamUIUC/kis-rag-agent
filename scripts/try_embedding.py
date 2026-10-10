"""
실험: 임베딩이 뭘 하는지 눈으로 보기

실행 (프로젝트 루트에서):
    pip install numpy sentence-transformers
    python scripts/try_embedding.py

처음 실행하면 bge-m3 모델(약 2.3GB)을 내려받아서 몇 분 걸린다. 두 번째부터는 바로 뜬다.
"""

import time

import torch
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------------
# 0. 모델 불러오기
# ------------------------------------------------------------------
t0 = time.time()
model = SentenceTransformer("BAAI/bge-m3", device="cuda")
model.half()                    # 16비트로: 메모리 절반, 속도 향상 (임베딩 품질 차이는 거의 없음)
model.max_seq_length = 1024     # 청크 최대 1,500자 ≈ 1,000토큰 이하라 이 정도면 충분
print(f"모델 로딩 {time.time() - t0:.1f}초 | GPU 메모리 {torch.cuda.memory_allocated() / 1e9:.2f} GB")


def embed(texts):
    """글자 리스트 -> 벡터 행렬 (글자 수 x 1024). 길이 1로 정규화해서 내적 = 코사인 유사도."""
    return model.encode(texts, normalize_embeddings=True, convert_to_tensor=True)


def show(title, queries, docs):
    q, d = embed(queries), embed(docs)
    scores = q @ d.T            # 행렬곱 한 번 = 모든 (질문, 문서) 쌍의 유사도
    print(f"\n=== {title} ===")
    for i, query in enumerate(queries):
        print(f"질문: {query}")
        for j in scores[i].argsort(descending=True).tolist():
            print(f"   {scores[i, j].item():.3f}  {docs[j]}")


# ------------------------------------------------------------------
# 1. 벡터는 어떻게 생겼나
# ------------------------------------------------------------------
v = embed(["영업이익이 크게 감소했다"])[0]
print(f"\n=== 1. 벡터의 모양 ===\n문장 하나 -> 숫자 {v.shape[0]}개")
print("앞 8개:", [round(x, 3) for x in v[:8].float().tolist()])
print("길이(노름):", round(v.float().norm().item(), 3), "← 정규화해서 항상 1")

# ------------------------------------------------------------------
# 2. 단어가 달라도 뜻이 비슷하면 가깝다
# ------------------------------------------------------------------
show(
    "2. 뜻이 비슷하면 점수가 높다",
    ["영업이익이 줄어든 이유"],
    [
        "수익성이 악화되었습니다",                 # 단어는 다르지만 뜻이 비슷
        "Operating profit declined sharply",       # 영어지만 뜻이 비슷
        "영업이익이 전년 대비 증가하였습니다",      # 단어는 겹치지만 뜻은 반대
        "당사 본사는 경기도 수원시에 있습니다",     # 무관
    ],
)

# ------------------------------------------------------------------
# 3. 헤더가 왜 필요한가 (embed_text vs text)
# ------------------------------------------------------------------
body = "원재료 가격 상승으로 매출원가가 증가하였습니다."
show(
    "3-1. 헤더 없이 (text만): 어느 회사 청크인지 구분 못 함",
    ["삼성전자 원재료 가격 상승 영향"],
    [body, body],  # 두 회사의 본문이 똑같다고 가정
)
show(
    "3-2. 헤더 붙여서 (embed_text): 삼성전자 청크가 위로",
    ["삼성전자 원재료 가격 상승 영향"],
    [
        f"[LG전자 | 사업보고서 (2025.12) | II. 사업의 내용 > 3. 원재료]\n{body}",
        f"[삼성전자 | 사업보고서 (2025.12) | II. 사업의 내용 > 3. 원재료]\n{body}",
    ],
)

# ------------------------------------------------------------------
# 4. 속도: 실제 검색은 행렬곱 한 번
# ------------------------------------------------------------------
fake_db = torch.nn.functional.normalize(
    torch.randn(50_000, 1024, device="cuda", dtype=torch.float16), dim=1
)  # 청크 5만 개 크기의 가짜 벡터 묶음
q = embed(["삼성전자 원재료 가격 상승 영향"])
torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    top = (q @ fake_db.T).topk(5)
torch.cuda.synchronize()
print(f"\n=== 4. 검색 속도 ===\n청크 5만 개에서 상위 5개 찾기: 한 번에 {(time.time() - t0) * 10:.2f} ms")