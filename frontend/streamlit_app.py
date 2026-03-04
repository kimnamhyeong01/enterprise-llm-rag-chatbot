import base64
import os
import uuid

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

DOMAIN_LABELS = {
    "manual": "🔧 현장 매뉴얼",
    "hr": "👤 사내 규정/HR",
    "it": "💻 IT 헬프데스크",
    "sales": "📊 영업/고객 대응",
    "education": "📚 교육/온보딩",
}

DOMAIN_COLORS = {
    "manual": "#FF6B35",
    "hr": "#4ECDC4",
    "it": "#45B7D1",
    "sales": "#96CEB4",
    "education": "#FFEAA7",
}

st.set_page_config(
    page_title="전사 RAG 챗봇",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 전사 업무 지원 RAG 챗봇")
st.caption("현장 매뉴얼 · HR · IT · 영업 · 교육 도메인 자동 라우팅")

# 세션 초기화
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "chat_messages" not in st.session_state:
    # 각 메시지: {"role": str, "content": str, "domain": str, "sources": list}
    st.session_state.chat_messages = []

# 사이드바
with st.sidebar:
    st.subheader("📌 지원 도메인")
    for domain, label in DOMAIN_LABELS.items():
        st.markdown(f"**{label}**")

    st.divider()

    st.subheader("⚙️ 세션 관리")
    st.caption(f"Thread ID: `{st.session_state.thread_id[:8]}...`")

    if st.button("🔄 새 대화 시작", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat_messages = []
        st.rerun()

    st.divider()
    st.caption(f"Backend: `{BACKEND_URL}`")

    # 이미지 업로드 (현장 매뉴얼 도메인 활용)
    st.subheader("📷 이미지 첨부")
    st.caption("현장 문제 상황 사진을 첨부하면 AI가 분석합니다.")
    uploaded_image = st.file_uploader(
        "이미지 선택",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    if uploaded_image:
        st.image(uploaded_image, caption="첨부된 이미지", use_container_width=True)

# 기존 대화 출력
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            domain = msg.get("domain", "")
            if domain:
                label = DOMAIN_LABELS.get(domain, domain)
                color = DOMAIN_COLORS.get(domain, "#999")
                st.markdown(
                    f'<span style="background-color:{color};color:#333;'
                    f'padding:2px 8px;border-radius:4px;font-size:12px;'
                    f'font-weight:bold;">{label}</span>',
                    unsafe_allow_html=True,
                )
            st.markdown(msg["content"])

            sources = msg.get("sources", [])
            if sources:
                with st.expander("📄 참고 문서"):
                    for src in sources:
                        st.markdown(f"- `{src}`")
        else:
            st.markdown(msg["content"])

# 채팅 입력
user_input = st.chat_input("질문을 입력하세요. (예: 연차 이월 규칙이 뭐야?)")

if user_input:
    # 이미지 인코딩
    image_base64 = None
    image_media_type = None
    if uploaded_image:
        uploaded_image.seek(0)
        raw_bytes = uploaded_image.read()
        image_base64 = base64.b64encode(raw_bytes).decode("utf-8")
        image_media_type = uploaded_image.type or "image/jpeg"

    # 사용자 메시지 표시
    display_text = user_input
    if uploaded_image:
        display_text += f"\n\n📷 *이미지 첨부: {uploaded_image.name}*"

    st.session_state.chat_messages.append({"role": "user", "content": display_text})
    with st.chat_message("user"):
        st.markdown(display_text)

    # 백엔드 호출
    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={
                        "thread_id": st.session_state.thread_id,
                        "user_input": user_input,
                        "image_base64": image_base64,
                        "image_media_type": image_media_type,
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                reply = data.get("reply", "(응답 없음)")
                domain = data.get("domain", "")
                sources = data.get("sources", [])
            except Exception as e:
                reply = f"⚠️ 백엔드 호출 중 오류가 발생했습니다: {e}"
                domain = ""
                sources = []

        # 도메인 배지 표시
        if domain:
            label = DOMAIN_LABELS.get(domain, domain)
            color = DOMAIN_COLORS.get(domain, "#999")
            st.markdown(
                f'<span style="background-color:{color};color:#333;'
                f'padding:2px 8px;border-radius:4px;font-size:12px;'
                f'font-weight:bold;">{label}</span>',
                unsafe_allow_html=True,
            )

        st.markdown(reply)

        if sources:
            with st.expander("📄 참고 문서"):
                for src in sources:
                    st.markdown(f"- `{src}`")

    st.session_state.chat_messages.append(
        {
            "role": "assistant",
            "content": reply,
            "domain": domain,
            "sources": sources,
        }
    )
