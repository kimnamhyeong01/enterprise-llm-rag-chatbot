from langchain_openai import ChatOpenAI

from .config import OPENAI_API_KEY, OPENAI_MODEL


DOMAIN_DESCRIPTIONS = {
    "manual": "설비, 장비, 공정, 압력 알람, 에러 코드, 필터 교체, 라인 운영, 안전 규정, 트러블슈팅",
    "hr": "인사, 복지, 출장비, 연차, 휴가, 재택근무, 근태, 경비 처리, 승진, 평가",
    "it": "IT 장애, VPN, SAP, 이메일, 시스템 오류, 네트워크, 프린터, 원격 접속",
    "sales": "제품 정보, 영업, 고객 대응, 경쟁사 비교, 제안서, 가격, A제품, B제품",
    "education": "교육, 온보딩, 공정 흐름, 품질 기준, 신입, 사내 시스템, 불량 처리",
}

_ROUTER_SYSTEM = """당신은 사용자 질문을 아래 카테고리 중 하나로 분류하는 전문가입니다.

카테고리 목록:
- manual: {manual}
- hr: {hr}
- it: {it}
- sales: {sales}
- education: {education}

반드시 카테고리 키워드(manual, hr, it, sales, education) 중 하나만 출력하세요.
다른 내용은 절대 출력하지 마세요.""".format(**DOMAIN_DESCRIPTIONS)

_router_llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model=OPENAI_MODEL,
    temperature=0,
)


def classify_domain(query: str) -> str:
    """사용자 질문을 도메인으로 분류한다."""
    messages = [
        ("system", _ROUTER_SYSTEM),
        ("human", query),
    ]
    response = _router_llm.invoke(messages)
    domain = response.content.strip().lower()

    valid_domains = set(DOMAIN_DESCRIPTIONS.keys())
    if domain not in valid_domains:
        # 키워드 기반 폴백 분류
        domain = _keyword_fallback(query)

    return domain


def _keyword_fallback(query: str) -> str:
    """키워드 기반 폴백 도메인 분류."""
    q = query.lower()

    hr_keywords = ["연차", "휴가", "출장", "재택", "근태", "복지", "인사", "경조", "육아", "병가", "급여"]
    it_keywords = ["vpn", "sap", "이메일", "메일", "프린터", "시스템", "로그인", "비밀번호", "원격"]
    sales_keywords = ["제품", "a제품", "b제품", "영업", "고객", "경쟁사", "가격", "제안서"]
    education_keywords = ["교육", "온보딩", "신입", "품질 검사", "공정 흐름", "불량"]

    if any(k in q for k in hr_keywords):
        return "hr"
    if any(k in q for k in it_keywords):
        return "it"
    if any(k in q for k in sales_keywords):
        return "sales"
    if any(k in q for k in education_keywords):
        return "education"
    return "manual"
