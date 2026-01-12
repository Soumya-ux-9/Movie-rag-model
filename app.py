
# app.py — Final Stable RAG

import os
import warnings
warnings.filterwarnings("ignore")

from transformers import pipeline
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline

# Paths

DATA_PATH = "movies.txt"
CHROMA_PATH = "chroma_db"


# Load documents
def load_docs():
    loader = TextLoader(DATA_PATH, encoding="utf-8")
    return loader.load()


# Split documents
def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_documents(docs)


# Embeddings

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# Vector DB

def get_db():
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embeddings()
    )


# Ingest data (run once)

def ingest():
    docs = load_docs()
    chunks = split_docs(docs)
    db = get_db()
    db.add_documents(chunks)
    db.persist()
    print("Documents ingested successfully")


# Lightweight CPU LLM

def get_llm():
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_length=128
    )
    return HuggingFacePipeline(pipeline=pipe)

def is_story_question(q):
    keywords = [
        "story", "plot", "about", "what happens",
        "ending", "explain", "summary"
    ]
    return any(k in q.lower() for k in keywords)


# Rule-based extractors
def extract_lead_actor(context):
    for line in context.split("\n"):
        if "Aamir Khan" in line:
            return "Aamir Khan"
    return None

def extract_who_dies(context):
    for line in context.split("\n"):
        l = line.lower()
        if "kalpana" in l and ("killed" in l or "dies" in l):
            return "Kalpana Shetty"
    return None

def extract_lead_actress(context):
    for line in context.split("\n"):
        if "Asin" in line:
            return "Asin"
    return None

def extract_director(context):
    for line in context.split("\n"):
        if "Director" in line:
            return line.split(":", 1)[1].strip()
    return None

from textblob import TextBlob

def movie_good_or_bad(context):
    polarity = TextBlob(context).sentiment.polarity

    if polarity > 0.1:
        return "The movie is generally considered GOOD."
    elif polarity < -0.1:
        return "The movie is generally considered BAD."
    else:
        return "The movie has MIXED reviews."
    
def extract_villain(context):
    for line in context.split("\n"):
        if "villain" in line.lower():
            return line.split(":", 1)[1].strip()
    return None


def extract_inspiration(context):
    for line in context.split("\n"):
        if "inspired" in line.lower() or "remake" in line.lower():
            return line.strip()
    return None


def extract_box_office(context):
    lines = context.split("\n")
    for i, line in enumerate(lines):
        if "box office collection" in line.lower():
            # If value is on same line
            if ":" in line and line.split(":", 1)[1].strip():
                return line.split(":", 1)[1].strip()
            # If value is on next line
            if i + 1 < len(lines):
                return lines[i + 1].strip()
    return None







# Ask question

def ask(question):
    db = get_db()
    docs = db.similarity_search(question, k=3)
    context = "\n".join(d.page_content for d in docs)

    # ---- STORY / PLOT QUESTIONS ----
    if is_story_question(question):
        prompt = f"""
Answer using ONLY the context.
Explain clearly in 4–5 lines.
If not found, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""
        llm = get_llm()
        print("\n Answer:\n", llm.invoke(prompt))
        return

    # ---- Deterministic factual handling ----
    if "lead actor" in question.lower():
        answer = extract_lead_actor(context)
        print("\n Answer:\n", answer if answer else "I don't know")
        return

    if "who dies" in question.lower() or "dies" in question.lower():
        answer = extract_who_dies(context)
        print("\n Answer:\n", answer if answer else "I don't know")
        return
    
    if "lead actress" in question.lower():
        answer = extract_lead_actress(context)
        print("\n Answer:\n", answer if answer else "I don't know")
        return
    
    if "director" in question.lower():
        answer = extract_director(context)
        print("\n Answer:\n", answer if answer else "I don't know")
        return
    
    
    if "good or bad" in question.lower() or "is the movie good" in question.lower():
        verdict = movie_good_or_bad(context)
        print("\n Answer:\n", verdict)
        return
    
    if "villain" in question.lower():
        ans = extract_villain(context)
        print("\n Answer:\n", ans if ans else "I don't know")
        return
    
    if "copied" in question.lower() or "inspired" in question.lower():
        ans = extract_inspiration(context)
        print("\n Answer:\n", ans if ans else "I don't know")
        return
    
    if "collection" in question.lower() or "box office" in question.lower():
        ans = extract_box_office(context)
        print("\n Answer:\n", ans if ans else "I don't know")
        return






    # ---- LLM fallback for descriptive questions ----
    prompt = f"""
Answer ONLY using the context.
Give a short direct answer.
If not found, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""

    llm = get_llm()
    print("\n Answer:\n", llm.invoke(prompt))



# Main

if __name__ == "__main__":

    if not os.path.exists(CHROMA_PATH):
        ingest()

    print("\n Movie RAG System Ready")
    print("Type 'exit' to quit\n")

    while True:
        q = input("❓ Question (exit to quit): ")
        if q.lower() == "exit":
            break
        ask(q)
