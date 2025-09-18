from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings


# Step 2: 建立向量数据库 (以 FAISS + HuggingFace embedding 为例)
embeddings = HuggingFaceEmbeddings(
    model_name="/data-slow/llm-models/sentence-embedding-models/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda:2"},
    encode_kwargs={"normalize_embeddings": True},  # 常用设置，提升检索效果
)
vectorstore = FAISS.load_local(
    "/data-slow/knowledge-hub/QS_index",
    embeddings,
    allow_dangerous_deserialization=True,
)  # 假设你已经构建好了索引

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})


query = ["公司地址"]
results = retriever.get_relevant_documents(query=query[0])
for r in results:
    print(f"metadata:{r.metadata}")
    print(f"page_content:{r.page_content}")