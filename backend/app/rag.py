from pathlib import Path
from typing import Dict, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from .config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL


BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
CHROMA_DIR = BASE_DIR / "chroma_db"

DOMAINS = ["manual", "hr", "it", "sales", "education"]

_retrievers: Dict[str, object] = {}


def _get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        model=OPENAI_EMBEDDING_MODEL,
    )


def load_domain_documents(domain: str) -> List[Document]:
    """도메인 디렉토리의 모든 .txt 파일을 로드하고 청킹한다."""
    domain_dir = KNOWLEDGE_DIR / domain
    raw_texts: List[tuple[str, str]] = []

    if domain_dir.exists():
        for txt_file in sorted(domain_dir.glob("*.txt")):
            content = txt_file.read_text(encoding="utf-8")
            raw_texts.append((txt_file.name, content))

    if not raw_texts:
        return [
            Document(
                page_content=f"[{domain}] 관련 문서가 아직 준비되지 않았습니다.",
                metadata={"source": "default", "domain": domain},
            )
        ]

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    docs: List[Document] = []
    for filename, text in raw_texts:
        for chunk in splitter.split_text(text):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={"source": filename, "domain": domain},
                )
            )
    return docs


def get_domain_retriever(domain: str):
    """도메인별 벡터 검색기를 반환한다. 최초 호출 시 벡터 DB를 생성/로드한다."""
    if domain in _retrievers:
        return _retrievers[domain]

    embeddings = _get_embeddings()
    collection_name = f"rag_{domain}"

    # 기존 컬렉션 로드 시도
    try:
        vectordb = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )
        if vectordb._collection.count() > 0:
            _retrievers[domain] = vectordb.as_retriever(search_kwargs={"k": 3})
            return _retrievers[domain]
    except Exception:
        pass

    # 지식 문서로 새 컬렉션 생성
    docs = load_domain_documents(domain)
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(CHROMA_DIR),
    )
    _retrievers[domain] = vectordb.as_retriever(search_kwargs={"k": 3})
    return _retrievers[domain]
