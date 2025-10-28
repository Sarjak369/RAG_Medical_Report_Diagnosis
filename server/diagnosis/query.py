import os
import asyncio
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-medical-diagnosis")


load_dotenv()

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

embed_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

llm = ChatGroq(temperature=0, model="llama-3.3-70b-versatile")

prompt = PromptTemplate.from_template(
    """
    You are a medical assistant. Using only the provided context (portions of the user's report), produce:
    1) A concise probable diagnosis (1-2 lines)
    2) Key findings from the report (bullet points)
    3) Recommended next steps (tests/treatments) - label clearly as suggestions, not medical advice.
    
    Context:
    {context}
    
    User question:
    {questions}
    """
)

rag_chain = prompt | llm


async def diagnosis_report(user: str, doc_id: str, question: str):
    # embed the question
    question_embedding = await asyncio.to_thread(embed_model.embed_query, question)

    # pass query to pinecone to extract relevant info (retrieving top 5 relevant information)
    results = await asyncio.to_thread(index.query, vector=question_embedding, top_k=5, include_metadata=True)

    # filter for doc_id matches
    contexts = []
    sources_set = set()
    for match in results.get("matches", []):  # type:ignore
        md = match.get("metadata", {})
        if md.get("doc_id") == doc_id:
            # take text snippet
            text_snippet = md.get("text") or ""
            contexts.append(text_snippet)
            sources_set.add(md.get("source"))

    if not contexts:
        return {"diagnosis": None, "explanation": "No report content indexed for this doc_id"}

    # limit context length
    context_text = "/n/n".join(contexts[:5])

    # finally calling the rag chain
    final = await asyncio.to_thread(rag_chain.invoke, {"context": context_text, "questions": question})

    return {"diagnosis": final.content, "sources": list(sources_set)}
