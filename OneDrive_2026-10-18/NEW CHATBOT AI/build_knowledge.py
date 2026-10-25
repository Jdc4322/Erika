"""Construye una base de conocimiento local para física usando Chroma + sentence-transformers.

Uso básico:
  python build_knowledge.py --add-url "https://arxiv.org/abs/quant-ph/010101"
  python build_knowledge.py --ingest-dir ./pdfs
  python build_knowledge.py --sample
"""
from __future__ import annotations

import os
import argparse
from typing import List

import requests
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from chromadb import PersistentClient
from uuid import uuid4
from PyPDF2 import PdfReader

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "physics"


def init_chroma(model_name: str = "all-MiniLM-L6-v2"):
    client = chromadb.PersistentClient(CHROMA_DIR)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
    collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)
    return client, collection


def ingest_texts(collection, texts: List[str], metadatas: List[dict] | None = None):
    ids = [str(uuid4()) for _ in texts]
    collection.add(documents=texts, metadatas=metadatas or [{} for _ in texts], ids=ids)


def fetch_arxiv_abstract(url: str) -> str:
    # Simple fetch: for arXiv, use the abstract page scraping
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    text = r.text
    # crude extraction
    start = text.find('<meta name="citation_abstract" content="')
    if start != -1:
        start += len('<meta name="citation_abstract" content="')
        end = text.find('"', start)
        return requests.utils.unquote(text[start:end])
    # fallback: return title
    return url


def ingest_url(collection, url: str):
    try:
        if "arxiv.org/abs" in url:
            txt = fetch_arxiv_abstract(url)
            ingest_texts(collection, [txt], metadatas=[{"source": url}])
            return True
        # generic web page
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(r.text, "html.parser")
        paragraphs = [p.get_text(separator=" ") for p in soup.find_all("p")]
        text = "\n\n".join(paragraphs)
        if text.strip():
            ingest_texts(collection, [text], metadatas=[{"source": url}])
            return True
    except Exception:
        return False


def ingest_pdf(collection, path: str):
    try:
        reader = PdfReader(path)
        pages = []
        for p in reader.pages:
            pages.append(p.extract_text() or "")
        text = "\n\n".join(pages)
        ingest_texts(collection, [text], metadatas=[{"source": path}])
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--add-url", help="Añadir una URL a la KB")
    parser.add_argument("--ingest-dir", help="Ingesta todos los PDFs de un directorio")
    parser.add_argument("--sample", action="store_true", help="Agregar ejemplos de referencia (abstracts famosos)")
    args = parser.parse_args()

    client, collection = init_chroma()

    if args.sample:
        samples = [
            ("Schrodinger 1926 abstract", "On the present situation in quantum mechanics. Erwin Schrödinger, 1926."),
            ("Einstein 1905 abstract", "On the electrodynamics of moving bodies. Albert Einstein, 1905."),
        ]
        texts = [s[1] for s in samples]
        metas = [{"title": s[0]} for s in samples]
        ingest_texts(collection, texts, metas)
        print("Muestras añadidas.")

    if args.add_url:
        ok = ingest_url(collection, args.add_url)
        print("URL procesada:" , ok)

    if args.ingest_dir:
        for fn in tqdm(os.listdir(args.ingest_dir)):
            if fn.lower().endswith('.pdf'):
                path = os.path.join(args.ingest_dir, fn)
                ingest_pdf(collection, path)
        print("Ingesta de directorio finalizada.")


if __name__ == "__main__":
    main()
