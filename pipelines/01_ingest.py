import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.schema import Document

# -------------------------------
# 환경설정
# -------------------------------
VECTOR_DIR = os.getenv("VECTOR_DIR", "vectorstore")
DOCS_DIRS = ["docs/manual", "docs/qna"]
META_PATH = Path("data/artifacts/index_meta.json")
LOW_SCORE_FILE = "logs/low_score.json"

# ✅ bge-m3 임베딩
def get_embeddings():
    return HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-m3",
        encode_kwargs={"normalize_embeddings": True}
    )

# ✅ 문서 로딩
def load_documents():
    documents = []
    for docs_dir in DOCS_DIRS:
        if not os.path.isdir(docs_dir):
            continue
        for filename in os.listdir(docs_dir):
            file_path = os.path.join(docs_dir, filename)
            manual_name = os.path.splitext(filename)[0]

            if filename.endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = manual_name
                documents.extend(docs)

            elif filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                for i, page in enumerate(pages):
                    page.metadata["source"] = manual_name
                    page.metadata["page"] = i + 1
                    page.metadata["citation"] = f"{manual_name} {i+1}페이지"
                    documents.append(page)

    return documents

# ✅ low_score.json 로딩
def load_low_score_docs():
    docs = []
    if os.path.exists(LOW_SCORE_FILE):
        try:
            with open(LOW_SCORE_FILE, "r", encoding="utf-8") as f:
                low_score_data = json.load(f)
                for item in low_score_data:
                    q = item.get("question", "").strip()
                    a = item.get("expected", "").strip()
                    if q and a:
                        docs.append(Document(
                            page_content=f"Q: {q}\nA: {a}",
                            metadata={"source": "low_score_feedback"}
                        ))
            print(f"⚠️ low_score.json 반영됨 (총 {len(docs)}개)")
        except Exception as e:
            print(f"❌ low_score.json 로드 실패: {e}")
    return docs

# ✅ fingerprint 계산
def calc_fingerprint(docs):
    m = hashlib.md5()
    for d in docs:
        m.update(d.page_content.encode("utf-8"))
    return m.hexdigest()

# ✅ 인덱스 생성/갱신
def build_vectorstore(documents, low_score_docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = splitter.split_documents(documents)
    split_docs = [doc for doc in split_docs if len(doc.page_content.strip()) > 10]

    embeddings = get_embeddings()
    os.makedirs(VECTOR_DIR, exist_ok=True)

    index_path = os.path.join(VECTOR_DIR, "index.faiss")

    if os.path.exists(index_path):
        vectorstore = FAISS.load_local(
            VECTOR_DIR, embeddings, allow_dangerous_deserialization=True
        )
        print("📂 기존 벡터스토어 로드 완료")

        if split_docs:
            vectorstore.add_documents(split_docs)
            print(f"➕ 새 문서 {len(split_docs)}개 추가 임베딩 완료")

        if low_score_docs:
            vectorstore.add_documents(low_score_docs)
            print(f"➕ low_score {len(low_score_docs)}개 추가 임베딩 완료")

    else:
        vectorstore = FAISS.from_documents(split_docs + low_score_docs, embeddings)
        print("🆕 신규 벡터스토어 생성")

    vectorstore.save_local(VECTOR_DIR)
    print(f"✅ Vectorstore 저장 완료: {VECTOR_DIR}")
    return vectorstore, split_docs

# ✅ 메인 실행
if __name__ == "__main__":
    print("🚀 문서/low_score 로딩 시작...")

    docs = load_documents()
    low_score_docs = load_low_score_docs()
    print(f"📄 로드된 문서 수: {len(docs)}, low_score 문서 수: {len(low_score_docs)}")

    print("🔎 벡터스토어 생성/갱신 중...")
    result = build_vectorstore(docs, low_score_docs)

    if result:
        vectorstore, split_docs = result
        META_PATH.parent.mkdir(parents=True, exist_ok=True)
        fp = calc_fingerprint(split_docs)
        META_PATH.write_text(
            json.dumps({
                "docs_fingerprint": fp,
                "built_at": datetime.utcnow().isoformat()
            }, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"✅ index_meta.json 저장됨 (docs_fingerprint={fp})")
