import os
import json
import hashlib
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# -------------------------------
# 환경설정
# -------------------------------
VECTOR_DIR = os.getenv("VECTOR_DIR", "vectorstore")
DOCS_DIRS = ["docs/manual", "docs/qna"]
META_PATH = Path("data/artifacts/index_meta.json")


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
                loader = TextLoader(file_path, encoding='utf-8')
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
                    citation = f"\n\n(출처: {manual_name} {i+1}페이지)"
                    page.page_content += citation
                    documents.append(page)
    return documents


# ✅ fingerprint 계산
def calc_fingerprint(docs):
    m = hashlib.md5()
    for d in docs:
        m.update(d.page_content.encode("utf-8"))
    return m.hexdigest()


# ✅ 인덱스 생성
def build_vectorstore(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = splitter.split_documents(documents)
    split_docs = [doc for doc in split_docs if len(doc.page_content.strip()) > 10]

    if not split_docs:
        print("❌ 인덱싱할 문서가 없습니다.")
        return None

    os.makedirs(VECTOR_DIR, exist_ok=True)
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(VECTOR_DIR)
    print(f"✅ Vectorstore 저장 완료: {VECTOR_DIR}")
    return vectorstore, split_docs


# ✅ 메인 실행
if __name__ == "__main__":
    print("🚀 문서 로딩 시작...")
    docs = load_documents()
    print(f"📄 로드된 문서 페이지 수: {len(docs)}")

    print("🔎 벡터스토어 생성 중...")
    result = build_vectorstore(docs)

    if result:
        vectorstore, split_docs = result
        # fingerprint 기록
        META_PATH.parent.mkdir(parents=True, exist_ok=True)
        fp = calc_fingerprint(split_docs)
        META_PATH.write_text(
            json.dumps({"fingerprint": fp}, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"✅ index_meta.json 저장됨 (fingerprint={fp})")
