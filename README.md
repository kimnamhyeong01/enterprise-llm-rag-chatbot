# 전사 업무 지원 Multi-Domain RAG 챗봇 서비스

## 프로젝트 개요

전사 직원이 업무 중 발생하는 다양한 질문을 AI가 즉시 응답하는 **RAG(Retrieval-Augmented Generation) 기반 챗봇 서비스**다. 현장 매뉴얼, 사내 규정/HR, IT 헬프데스크, 영업/고객 대응, 교육/온보딩 등 **5개 업무 도메인**을 자동 라우팅하여 도메인별 사내 지식 문서를 기반으로 정확한 답변을 제공한다.

---

## 해결한 문제

| 기존 문제 | 해결 방안 |
|-----------|-----------|
| 업무 질문마다 담당 부서 문의 → 응답 지연 | AI가 즉시 자동 응답 |
| 매뉴얼/규정 문서가 방대하여 직접 검색 어려움 | RAG로 관련 문서만 발췌해 답변 생성 |
| 도메인별 담당자가 달라 어디에 물어볼지 모름 | LLM이 질문 의도를 파악해 자동 도메인 라우팅 |
| 현장 작업자가 설비 사진으로 문제 설명 불가 | 이미지 업로드 → 멀티모달 AI 분석 지원 |
| 지식 문서 업데이트 시 즉각 반영 불가 | 벡터 DB 재색인으로 즉시 반영 가능한 구조 |

---

## 지원 도메인

| 도메인 | 설명 | 지식 문서 |
|--------|------|-----------|
| 🔧 현장 매뉴얼 | 설비·공정·알람 조치, 에러 코드, 필터 교체 | `equipment_manual.txt`, `safety_manual.txt`, `troubleshooting.txt` |
| 👤 사내 규정/HR | 연차·출장비·재택근무 등 인사/복지 정책 | `hr_policy.txt`, `expense_policy.txt`, `vacation_policy.txt` |
| 💻 IT 헬프데스크 | VPN·SAP·이메일 등 IT 장애 대응 | `vpn_troubleshooting.txt`, `email_issue.txt` |
| 📊 영업/고객 대응 | 제품 비교, 경쟁사 분석, 고객 응대 전략 | `product_info.txt`, `competitor_analysis.txt` |
| 📚 교육/온보딩 | 공정 흐름, 품질 기준, 신입 교육 | `training_manual.txt`, `process_guide.txt` |

---

## 시스템 아키텍처

```
사용자 질문 (텍스트 + 선택적 이미지)
        │
        ▼
  [Streamlit 프론트엔드]
  - 도메인 배지 표시
  - 참고 문서 출처 표시
  - 이미지 업로드 (Base64 인코딩)
        │ POST /chat
        ▼
  [FastAPI 백엔드]
        │
        ▼
  [LangGraph 파이프라인]
  ┌─────────────────────────────┐
  │ router_node                 │
  │  LLM 기반 도메인 분류        │
  │  + 키워드 폴백 분류           │
  └────────────┬────────────────┘
               │ domain 결정
               ▼
  ┌─────────────────────────────┐
  │ retriever_node              │
  │  Chroma 벡터 DB 검색         │
  │  도메인별 컬렉션에서 Top-K 문서│
  └────────────┬────────────────┘
               │ context + sources
               ▼
  ┌─────────────────────────────┐
  │ generator_node              │
  │  도메인 전용 System Prompt   │
  │  LLM 답변 생성               │
  └────────────┬────────────────┘
               │
               ▼
  {reply, domain, sources} → 프론트엔드 응답
```

---

## 기술 스택

### 백엔드
| 구성 요소 | 기술 | 역할 |
|-----------|------|------|
| API 서버 | FastAPI | REST API 서빙, `/chat`, `/health` 엔드포인트 |
| AI 오케스트레이션 | LangGraph | 3-노드 상태 그래프 파이프라인 |
| LLM | OpenAI gpt-4o-mini | 도메인 분류 + 답변 생성 |
| 임베딩 | OpenAI text-embedding-3-small | 문서 벡터화 |
| 벡터 DB | Chroma | 도메인별 컬렉션 관리, 유사 문서 검색 |
| RAG 프레임워크 | LangChain | 문서 로드, 청킹, 검색기 구성 |

### 프론트엔드
| 구성 요소 | 기술 | 역할 |
|-----------|------|------|
| UI 프레임워크 | Streamlit | 채팅 인터페이스 구현 |
| 이미지 처리 | Base64 인코딩 | 현장 사진 멀티모달 전송 |

### 인프라
| 구성 요소 | 기술 | 역할 |
|-----------|------|------|
| 컨테이너화 | Docker + Docker Compose | 서비스 패키징 및 오케스트레이션 |
| 런타임 | Python 3.11 | 백엔드/프론트엔드 공통 |

---

## 핵심 구현 상세

### 1. Query Router (LLM + 키워드 폴백)

사용자 질문을 5개 도메인 중 하나로 분류한다. LLM이 1차 분류를 수행하고, 유효하지 않은 응답이 반환되면 키워드 매칭으로 폴백한다.

```python
# router.py
def classify_domain(query: str) -> str:
    response = _router_llm.invoke([("system", _ROUTER_SYSTEM), ("human", query)])
    domain = response.content.strip().lower()
    if domain not in valid_domains:
        domain = _keyword_fallback(query)  # 폴백
    return domain
```

### 2. LangGraph 상태 관리 (ChatState)

```python
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]  # 대화 이력
    domain: str      # 분류된 도메인 키
    context: str     # 검색된 문서 텍스트
    sources: List[str]  # 참고 문서 파일명 목록
```

### 3. 멀티 도메인 Chroma 벡터 DB

도메인별 독립 컬렉션(`rag_manual`, `rag_hr`, `rag_it`, `rag_sales`, `rag_education`)을 구성하고, 최초 호출 시 지식 문서를 로드·청킹(chunk_size=500, overlap=80)하여 벡터 DB를 생성한다. 이후 호출에서는 메모리 캐시에서 검색기를 반환한다.

### 4. 지식 문서 구성

```
backend/knowledge/
├── manual/   equipment_manual.txt, safety_manual.txt, troubleshooting.txt
├── hr/       hr_policy.txt, expense_policy.txt, vacation_policy.txt
├── it/       vpn_troubleshooting.txt, email_issue.txt
├── sales/    product_info.txt, competitor_analysis.txt
└── education/ training_manual.txt, process_guide.txt
```

### 5. 이미지 첨부 (멀티모달)

현장 작업자가 설비 사진을 업로드하면 Base64로 인코딩하여 백엔드에 전달하고, LLM이 이미지와 텍스트를 함께 분석하여 답변을 생성한다.

---

## API 명세

### POST /chat
```json
// 요청
{
  "thread_id": "uuid",
  "user_input": "3번 라인 압력 알람 조치 방법",
  "image_base64": "optional_base64_string",
  "image_media_type": "image/jpeg"
}

// 응답
{
  "reply": "3번 라인에서 압력 이상 알람이 발생하면...",
  "domain": "manual",
  "sources": ["equipment_manual.txt", "troubleshooting.txt"]
}
```

### GET /health
```json
{"status": "ok"}
```

---

## 실행 방법

```bash
# 1. 환경 변수 설정
export OPENAI_API_KEY=your_api_key

# 2. Docker Compose로 전체 서비스 실행
docker-compose up --build

# 3. 브라우저 접속
# 프론트엔드: http://localhost:8501
# 백엔드 API: http://localhost:8000
```

---

## 주요 특징 요약

- **자동 도메인 라우팅**: 사용자가 도메인을 선택하지 않아도 LLM이 질문 의도를 파악하여 적절한 RAG 파이프라인으로 자동 연결
- **대화 컨텍스트 유지**: Thread ID 기반 세션 관리로 멀티턴 대화 지원
- **참고 문서 출처 표시**: 모든 응답에 답변 근거가 된 지식 문서 파일명 제공
- **멀티모달 지원**: 텍스트 질문과 현장 사진을 함께 분석
- **확장 용이성**: 새 도메인 추가 시 knowledge 디렉토리에 txt 파일만 추가하면 자동 색인
- **Docker 기반 배포**: 단일 명령으로 전체 서비스 실행 가능
