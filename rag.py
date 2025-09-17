from langchain.chains import ConversationalRetrievalChain, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import (
    ChatOpenAI,
)  # 如果 sglang 提供 OpenAI API 兼容，可以直接用
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import sys

from langchain.callbacks.base import BaseCallbackHandler


class PrintHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs):
        print(token, end="", flush=True)


# Step 1: 定义 LLM (替换成你自己的 sglang endpoint)
llm = ChatOpenAI(
    model="deepseek-r1",
    openai_api_base="http://localhost:30000/v1",  # sglang 的 API 地址
    openai_api_key="none",  # 如果不需要认证，填个 dummy 即可
    streaming=True,  # 开启流式返回
    callbacks=[PrintHandler()],  # 添加回调处理器
)

# Step 2: 建立向量数据库 (以 FAISS + HuggingFace embedding 为例)
embeddings = HuggingFaceEmbeddings(
    model_name="/data-slow/llm-models/sentence-embedding-models/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda:2"},
    encode_kwargs={"normalize_embeddings": True},  # 常用设置，提升检索效果
)
vectorstore = FAISS.load_local(
    "/data-slow/knowledge-hub/QS_index", embeddings, allow_dangerous_deserialization=True
)  # 假设你已经构建好了索引
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Step 3: 定义记忆
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Step 4: 构建 RAG 对话链
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)


while True:
    try:
        # 添加提示符，等待用户输入
        question = input("User: ")

        # 移除行首尾的空格
        question = question.strip()

        # 检查是否退出
        if question.lower() in {"exit", "quit"}:
            print("Exiting...")
            break

        print("AI:", end="", flush=True)
        qa_chain.invoke(question)
        print("")  # 换行

    except EOFError:
        # 处理在命令行中按 Ctrl+D (Unix/Linux) 或 Ctrl+Z (Windows) 的情况
        print("\nExiting due to EOF.")
        break
