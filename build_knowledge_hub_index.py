from pypdf import PdfReader
from glob import glob
from langchain.text_splitter import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS

from langchain.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm


def main():

    data_regex = "/data-slow/knowledge-hub/QS/**/*.pdf"
    # model = SentenceTransformer(
    #     "/data-slow/llm-models/sentence-embedding-models/all-MiniLM-L6-v2"
    # )

    """
    hf_embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}  # 或 "cuda" 使用 GPU
    )
    """
    hf_embedding = HuggingFaceEmbeddings(
        model_name="/data-slow/llm-models/sentence-embedding-models/all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},  # 常用设置，提升检索效果
    )

    vectorstore = None

    with tqdm(glob(data_regex), desc="processing files") as pbar_files:

        for file in pbar_files:
            reader = PdfReader(file)
            splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)

            fname = file.split("knowledge-hub/")[1]

            with tqdm(
                enumerate(reader.pages), desc=f"  - Handling {fname}", leave=False
            ) as pbar_data:

                for i, page in pbar_data:
                    text = page.extract_text()
                    page_chunks = splitter.split_text(text)
                    # 带上页码 metadata

                    if len(page_chunks) == 0:
                        continue

                    page_chunks = [
                        f"以下是文件名为 {fname} 中的内容: {chunk}"
                        for chunk in page_chunks
                    ]

                    batch_vs = FAISS.from_texts(
                        texts=page_chunks,
                        embedding=hf_embedding,
                        metadatas=[{"page": i + 1, "fname": fname}] * len(page_chunks),
                    )

                    if vectorstore is None:
                        vectorstore = batch_vs

                    else:
                        vectorstore = vectorstore.merge_from(batch_vs)

    vectorstore.save_local("/data-slow/knowledge-hub/QS_index")


if __name__ == "__main__":
    main()
