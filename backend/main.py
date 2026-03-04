import base64
from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_core.messages import HumanMessage, AIMessage

from app.graph import graph


class ChatRequest(BaseModel):
    thread_id: str
    user_input: str
    image_base64: Optional[str] = None
    image_media_type: Optional[str] = "image/jpeg"


class ChatResponse(BaseModel):
    reply: str
    domain: str
    sources: List[str]


app = FastAPI(title="LangGraph Multi-Domain RAG Backend")

origins = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # 이미지 포함 여부에 따라 HumanMessage 구성
    if req.image_base64:
        content = [
            {"type": "text", "text": req.user_input},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{req.image_media_type};base64,{req.image_base64}"
                },
            },
        ]
        human_msg = HumanMessage(content=content)
    else:
        human_msg = HumanMessage(content=req.user_input)

    result_state = graph.invoke(
        {
            "messages": [human_msg],
            "domain": "",
            "context": "",
            "sources": [],
        }
    )

    ai_msg = None
    for msg in reversed(result_state["messages"]):
        if isinstance(msg, AIMessage):
            ai_msg = msg
            break

    reply = ai_msg.content if ai_msg else "(응답이 없습니다.)"
    domain = result_state.get("domain", "")
    sources = result_state.get("sources", [])

    return ChatResponse(reply=reply, domain=domain, sources=sources)
