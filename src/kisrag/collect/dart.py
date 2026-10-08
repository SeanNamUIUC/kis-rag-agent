"""
Day 1: DART 공시 수집 스크립트
- corpCode.xml로 종목코드 -> DART 고유번호 매핑
- 종목별 공시 목록(list.json) 수집 -> data/meta/{ticker}.jsonl
- 정기공시/주요사항보고 원문(document.xml, zip) 다운로드 -> data/raw/{ticker}/{rcept_no}/

사용법 (프로젝트 루트 kis-rag-agent/ 에서 실행):
    pip install requests tqdm python-dotenv
    python src/kisrag/collect/dart.py --no-download   # 먼저 공시 건수만 확인
    python src/kisrag/collect/dart.py                 # 원문까지 다운로드
"""



"""
corp_codes.json	종목코드 → 고유번호	작음	수집할 때만
meta/	공시 목록 (제목, 날짜, 번호)	작음	문서 필터링, 출처 표시
raw/	공시 본문	큼	RAG의 핵심 재료
summary.json	건수 집계	아주 작음	확인용

"""
import argparse#명령어 option ex)--no-download 
import io #다운받은 데이터를 파일저장하지않고 메모리에서 처리
import json
import os
import time
import zipfile #zip파일 압축 해제
import xml.etree.ElementTree as ET #xml 파싱
from pathlib import Path

import requests # 인터넷 요청
from dotenv import load_dotenv # .env 파일 읽기
from tqdm import tqdm #진행률 표시

load_dotenv()  # 프로젝트 루트의 .env 파일에서 변수 다 가져옴.
API_KEY = os.environ.get("DART_API_KEY")
BASE = "https://opendart.fss.or.kr/api"
DATA = Path("data")
SLEEP = 0.3  # DART 호출 제한 대비 -> 너무 빨리 호출하면 429 Too Many Requests 에러 발생

# 섹터별로 고르게 선정한 20개 종목 (나중에 configs/tickers.yaml로 분리)
#종목코드 : 회사이름 dictionary
TICKERS = {
    "005930": "삼성전자",
    "000660": "SK하이닉스",
    "373220": "LG에너지솔루션",
    "207940": "삼성바이오로직스",
    "005380": "현대차",
    "000270": "기아",
    "035420": "NAVER",
    "035720": "카카오",
    "068270": "셀트리온",
    "105560": "KB금융",
    "055550": "신한지주",
    "005490": "POSCO홀딩스",
    "051910": "LG화학",
    "006400": "삼성SDI",
    "012330": "현대모비스",
    "028260": "삼성물산",
    "066570": "LG전자",
    "017670": "SK텔레콤",
    "003550": "LG",
    "071050": "한국금융지주",  # 한국투자증권 모회사 - 데모용
}

# 공시 유형
# A: 정기공시(사업/반기/분기보고서) -> 회사의 종합 성적표
# B: 주요사항보고 -> 회사의 중요한 결정 속보
# I: 거래소공시
DISCLOSURE_TYPES = ["A", "B", "I"]
# 원문까지 받을 유형 (I는 양이 많고 짧은 공시가 대부분이라 메타만)
DOWNLOAD_TYPES = {"A", "B"}




# 캐시 확인 → 없으면 zip 다운로드 → 메모리에서 압축 해제 → XML 파싱 → 상장사만 골라 딕셔너리 생성 → 파일로 저장 → 반환
def get_corp_codes() -> dict:
    """종목코드(6자리) -> DART 고유번호(8자리) 매핑. 한 번 받으면 캐시 사용."""
    cache = DATA / "corp_codes.json"
    #캐시 파일에 저장되어 있으면 캐시 사용
    if cache.exists():
        #문자열로 읽고  json.loads()로 dict로 변환 
        return json.loads(cache.read_text(encoding="utf-8"))


    #DART에서 corpCode.xml 다운로드 후 압축 해제
    r = requests.get(f"{BASE}/corpCode.xml", params={"crtfc_key": API_KEY}, timeout=60)
    r.raise_for_status()
    # 압축 해제 후 첫 번째 파일 읽기
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        xml_bytes = zf.read(zf.namelist()[0])

# XML 구조 예시

#     <result>
#   <list>
#     <corp_code>00126380</corp_code>
#     <corp_name>삼성전자</corp_name>
#     <stock_code>005930</stock_code>
#     <modify_date>20240101</modify_date>
#   </list>
#   <list>
#     <corp_code>00434003</corp_code>
#     <corp_name>어떤비상장회사</corp_name>
#     <stock_code> </stock_code>
#     ...
#   </list>
#   ... (약 10만 개)
# </result>

    root = ET.fromstring(xml_bytes) # result 태그를 root로 하는 ElementTree 객체 생성
    mapping = {}
    for item in root.iter("list"):
        stock = (item.findtext("stock_code") or "").strip() #상장사 -> 6자리 종목코드, 비상장사 -> 공백
        if stock:  # 상장사만
            # 종목코드(6자리) -> DART 고유번호(8자리) 매핑
            mapping[stock] = item.findtext("corp_code").strip()

    cache.parent.mkdir(parents=True, exist_ok=True)
    # json.dumps()로 dict를 json으로로 변환 후 cache 파일에 저장
    cache.write_text(json.dumps(mapping, ensure_ascii=False), encoding="utf-8")
    return mapping


#회사번호,시작일,종료일,공시유형 -> 공시목록 받기기
def list_disclosures(corp_code: str, start: str, end: str, pblntf_ty: str) -> list:
    """공시 목록을 페이지 단위(100건)로 끝까지 수집."""
    results, page = [], 1
    while True:
        params = {
            "crtfc_key": API_KEY,
            "corp_code": corp_code,
            "bgn_de": start,
            "end_de": end,
            "pblntf_ty": pblntf_ty, #공시 유형 (A: 정기공시, B: 주요사항보고, I: 거래소공시)
            "last_reprt_at": "Y",  # 정정 공시가 있으면 최종본만
            "page_no": page,
            "page_count": 100,
        }
        data = requests.get(f"{BASE}/list.json", params=params, timeout=30).json()
        time.sleep(SLEEP)


        # JSON 구조 예시
        # {
        # "status": "000",
        # "message": "정상",
        # "page_no": 1,
        # "total_page": 3,
        # "list": [
        #     {"corp_name": "삼성전자", "report_nm": "분기보고서 (2025.06)", "rcept_no": "20250814000123", "rcept_dt": "20250814", ...},
        #     ...
        # ]
        # }

        status = data.get("status")
        if status == "013":  # 조회 결과 없음
            break
        if status != "000":
            print(f"  [warn] {corp_code} {pblntf_ty}: {status} {data.get('message')}")
            break

        for row in data.get("list", []): #공시를 하나씩 꺼내서 
            row["pblntf_ty"] = pblntf_ty
            # 예시
            # {"report_nm": "분기보고서", "rcept_no": "20250814000123", ..., "pblntf_ty": "A"}
            results.append(row)

        if page >= int(data.get("total_page", 1)):
            break
        page += 1

    return results # 총 x건이 담긴 list 반환

#접수번호, 저장경로 -> 원문 zip 다운로드 후 압축 해제
def download_document(rcept_no: str, out_dir: Path) -> bool:
    """공시 원문 zip 다운로드 후 압축 해제. 이미 받았으면 건너뜀."""
    if out_dir.exists() and any(out_dir.iterdir()):#폴더안 항목들 하나씩 체크
        return True
    r = requests.get(
        f"{BASE}/document.xml",
        params={"crtfc_key": API_KEY, "rcept_no": rcept_no},
        timeout=60,
    )
    time.sleep(SLEEP)
    if not r.content.startswith(b"PK"):  # zip이 아니면 에러 응답(XML)
        print(f"  [warn] 원문 실패 {rcept_no}: {r.text[:120]}")
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        zf.extractall(out_dir)
    return True


def main():
    parser = argparse.ArgumentParser()
    #명령어 옵션 --start, --end, --no-download
    parser.add_argument("--start", default="20250101")
    parser.add_argument("--end", default=time.strftime("%Y%m%d"))  # 기본값: 오늘
    parser.add_argument("--no-download", action="store_true", help="메타데이터만 수집")
    args = parser.parse_args()

    if not API_KEY:
        raise SystemExit(".env 파일에 DART_API_KEY를 넣었는지 확인하세요.")

    corp_codes = get_corp_codes()
    (DATA / "meta").mkdir(parents=True, exist_ok=True)

    #마지막에 종목별 공시 건수와 원문 다운로드 건수를 summary.json으로 저장
    summary = {}
    for ticker, name in TICKERS.items():
        corp_code = corp_codes.get(ticker)
        if not corp_code:
            print(f"[skip] {name}({ticker}) 고유번호 없음")
            continue

        rows = []
        for ty in DISCLOSURE_TYPES:
            rows += list_disclosures(corp_code, args.start, args.end, ty)
        for row in rows:
            row["ticker"], row["name"] = ticker, name


        #메타데이터를 data/meta/{ticker}.jsonl로 저장 (한 줄에 한 건씩 JSON)
        meta_path = DATA / "meta" / f"{ticker}.jsonl"
        with meta_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        #원문 다운로드
        ok = 0
        if not args.no_download:
            # rows에서 유형이 A,B 인것만 새로운 리스트로 targets에 담음
            targets = [r for r in rows if r["pblntf_ty"] in DOWNLOAD_TYPES]
            for row in tqdm(targets, desc=name, leave=False):
                out = DATA / "raw" / ticker / row["rcept_no"]
                ok += download_document(row["rcept_no"], out)

        summary[name] = {"공시수": len(rows), "원문": ok}
        print(f"[done] {name}: 공시 {len(rows)}건, 원문 {ok}건")

    (DATA / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()