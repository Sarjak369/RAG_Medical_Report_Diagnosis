import os
import time
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from ..config.db import reports_collection
from typing import List
from fastapi import UploadFile

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag_medical_diagnosis")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)


# initialize pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud="aws", region=PINECONE_ENV)
existing_indexes = [i["name"] for i in pc.list_indexes()]

if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(name=PINECONE_INDEX_NAME, dimension=1536,
                    metric="dotproduct", spec=spec)
    # creating an index generally takes 1 or 2 secs of time. So let us add some relaxation time
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(1)

index = pc.Index(PINECONE_INDEX_NAME)


async def load_vectorstore(uploaded_files: List[UploadFile], uploaded: str, doc_id: str):
    """
        Save files, chunck texts, embed texts, upsert in Pinecone and write metadata to Mongo.
    """

    embed_model = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    for file in uploaded_files:
        if not file.filename:
            raise ValueError("One of the uploaded files has no filename!")

        if not doc_id:
            raise ValueError("doc_id cannot be None.")

        filename = Path(file.filename).name
        save_path = Path(UPLOAD_DIR) / f"{doc_id}_{filename}"
        content = await file.read()

        with open(save_path, "wb") as f:
            f.write(content)

        # load pdf pages
        loader = PyPDFLoader(str(save_path))
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100)

        chunks = splitter.split_documents(documents)

        texts = [chunk.page_content for chunk in chunks]
        ids = [f"{doc_id}-{i}" for i in range(len(chunks))]
        metadata = [
            {
                "source": filename,
                "doc_id": doc_id,
                "uploader": uploaded,
                "page": chunk.metadata.get("page", None),
                # store snippet in metadata (avoid huge fields)
                "text": chunk.page_content[:2000]
            }
            for chunk in chunks
        ]

        # get embeddings in thread
        embeddings = await asyncio.to_thread(embed_model.embed_documents, texts)

        # upsert - run in thread to avoid blocking
        def upsert():
            index.upsert(vectors=list(zip(ids, embeddings, metadata)))

        await asyncio.to_thread(upsert)

        # save report metadata in mongo
        reports_collection.insert_one({
            "doc_id": doc_id,
            "filename": filename,
            "uploader": uploaded,
            "num_chunks": len(chunks),
            "uploaded_at": time.time()
        })
