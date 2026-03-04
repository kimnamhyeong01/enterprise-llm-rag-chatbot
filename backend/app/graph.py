from typing import Annotated, List

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .config import OPENAI_API_KEY, OPENAI_MODEL
from .router import classify_domain
from .rag import get_domain_retriever


class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    domain: str
    context: str
    sources: List[str]


DOMAIN_LABELS = {
    "manual": "현장 매뉴얼 / 작업지시",
    "hr": "사내 규정 / HR",
    "it": "IT 헬프데스크",
    "sales": "영업 / 고객 대응",
    "education": "교육 / 온보딩",
}

_SYSTEM_PROMPT = """당신은 전사 업무 지원 AI 챗봇입니다. 담당 영역은 [{domain_label}]입니다.

아래 참고 문서를 바탕으로 사용자 질문에 정확하고 친절하게 답변하세요.

[참고 문서]
{context}

답변 지침:
- 참고 문서에 있는 내용을 기반으로 답변하세요.
- 단계별 절차가 있으면 번호 목록으로 안내하세요.
- 참고 문서에 없는 내용은 "관련 문서에서 해당 정보를 찾을 수 없습니다."라고 명확히 말하세요.
- 내선 번호, 링크 등 구체적인 정보가 있으면 함께 안내하세요.
- 한국어로 답변하세요."""

llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model=OPENAI_MODEL,
    temperature=0.3,
)


def _get_last_human_text(state: ChatState) -> str:
    """마지막 사용자 메시지의 텍스트를 추출한다."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            content = msg.content
            # 멀티모달 메시지(리스트) 처리
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return part["text"]
            return str(content)
    return ""


def router_node(state: ChatState) -> dict:
    """사용자 질문을 도메인으로 분류한다."""
    query = _get_last_human_text(state)
    domain = classify_domain(query)
    return {"domain": domain}


def retriever_node(state: ChatState) -> dict:
    """도메인별 RAG 검색으로 관련 문서를 가져온다."""
    query = _get_last_human_text(state)
    domain = state.get("domain", "manual")

    retriever = get_domain_retriever(domain)
    docs = retriever.get_relevant_documents(query)

    context_parts: List[str] = []
    sources: List[str] = []
    for i, doc in enumerate(docs, 1):
        context_parts.append(f"[문서 {i}]\n{doc.page_content.strip()}")
        src = doc.metadata.get("source", "unknown")
        if src not in sources:
            sources.append(src)

    context = "\n\n".join(context_parts) if context_parts else "관련 문서를 찾지 못했습니다."
    return {"context": context, "sources": sources}


def generator_node(state: ChatState) -> dict:
    """검색된 문서를 바탕으로 LLM이 최종 답변을 생성한다."""
    domain = state.get("domain", "manual")
    context = state.get("context", "관련 문서를 찾지 못했습니다.")
    domain_label = DOMAIN_LABELS.get(domain, domain)

    system_content = _SYSTEM_PROMPT.format(
        domain_label=domain_label,
        context=context,
    )

    messages = [SystemMessage(content=system_content)] + list(state["messages"])
    response = llm.invoke(messages)
    return {"messages": [response]}


def create_graph():
    builder = StateGraph(ChatState)

    builder.add_node("router", router_node)
    builder.add_node("retriever", retriever_node)
    builder.add_node("generator", generator_node)

    builder.add_edge(START, "router")
    builder.add_edge("router", "retriever")
    builder.add_edge("retriever", "generator")
    builder.add_edge("generator", END)

    return builder.compile()


graph = create_graph()
