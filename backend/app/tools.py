"""도메인별 RAG 검색 도구 (에이전트 기반 확장 시 활용 가능)"""
from langchain_core.tools import tool

from .rag import get_domain_retriever


def _rag_search(domain: str, query: str) -> str:
    retriever = get_domain_retriever(domain)
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return "관련 문서를 찾지 못했습니다."
    return "\n\n".join(
        f"[{i}] (출처: {d.metadata.get('source', '?')}) {d.page_content.strip()}"
        for i, d in enumerate(docs, 1)
    )


@tool
def manual_rag_search(query: str) -> str:
    """현장 매뉴얼, 설비 운영, 안전 규정, 에러 코드 관련 문서를 검색한다."""
    return _rag_search("manual", query)


@tool
def hr_rag_search(query: str) -> str:
    """사내 인사 규정, 출장비, 연차, 재택근무, 복지 관련 문서를 검색한다."""
    return _rag_search("hr", query)


@tool
def it_rag_search(query: str) -> str:
    """IT 장애 대응, VPN, SAP, 이메일 문제 관련 문서를 검색한다."""
    return _rag_search("it", query)


@tool
def sales_rag_search(query: str) -> str:
    """제품 정보, 영업 자료, 경쟁사 분석 관련 문서를 검색한다."""
    return _rag_search("sales", query)


@tool
def education_rag_search(query: str) -> str:
    """교육 자료, 공정 절차, 품질 기준 관련 문서를 검색한다."""
    return _rag_search("education", query)


TOOLS = [
    manual_rag_search,
    hr_rag_search,
    it_rag_search,
    sales_rag_search,
    education_rag_search,
]
