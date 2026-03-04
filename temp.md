# LangGraph RAG Service (FastAPI + Streamlit + Docker)

## 구조

- `backend/` : FastAPI(main.py) + LangGraph + RAG
- `frontend/` : Streamlit UI (백엔드 호출)
- `docker-compose.yml` : 두 서비스를 한 번에 실행

## 사용 방법

```bash
cd langgraph-rag-service

# env 파일 생성
# 프로젝트 루트 내 .env에 OPENAI_API_KEY 채우기
# frontend/.env의 BACKEND_URL은 docker-compose 기준으로 기본값 사용 (http://backend:8000)

# Docker 빌드 & 실행
docker-compose up --build #docker desktop 메뉴 내 container 화면에서 서비스 기동 안 된 경우 ▶️ 버튼 클릭 

# 삭제 필요 시 
docker-compose down
# Docker 
```

이후 브라우저에서 `http://localhost:8501` 접속.
