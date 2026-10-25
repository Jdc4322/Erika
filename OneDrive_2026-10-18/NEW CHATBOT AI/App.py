from __future__ import annotations

import json
import os
import random
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote_plus, unquote, urlparse, parse_qs

import truststore
truststore.inject_into_ssl()

import requests
import streamlit as st
import urllib3
from bs4 import BeautifulSoup

from dotenv import load_dotenv
import rag
import physics_response_utils
import local_image_generator
import local_attachments
import local_vision

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False

try:
    from serpapi import GoogleSearch
    SERPAPI_AVAILABLE = True
except ImportError:
    SERPAPI_AVAILABLE = False


DB_PATH = Path("chatbot_memory.db")
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
GEMINI_MODEL = "gemini-3.8-flash"
GEMINI_FALLBACK_MODEL = "gemini-3.5-flash-lite"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:streamGenerateContent?alt=sse"
DEFAULT_MODELS = ["llama3.2:latest", "mistral:latest", "llama3:latest", "gemma:latest"]
MODEL_KNOWLEDGE_NOTE = (
    "Responde con tu conocimiento local para preguntas estables. "
    "Usa fuentes de internet cuando se proporcionen o la pregunta dependa de informacion actual."
)
MAX_CONTEXT_MESSAGES = 4
MAX_MEMORIES = 4
MAX_WEB_RESULTS = 3
MAX_WEB_CHARS_PER_PAGE = 1000
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"
)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


@dataclass
class WebSource:
    title: str
    url: str
    text: str
    source: str


def db() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with db() as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL DEFAULT 'Nueva conversación',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )"""
        )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL, content TEXT NOT NULL,
                model TEXT, created_at TEXT NOT NULL,
                conversation_id INTEGER REFERENCES conversations(id)
            )"""
        )
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(messages)")}
        if "conversation_id" not in columns:
            connection.execute(
                "ALTER TABLE messages ADD COLUMN conversation_id INTEGER REFERENCES conversations(id)"
            )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL UNIQUE,
                source_message_id INTEGER, created_at TEXT NOT NULL
            )"""
        )
        orphan = connection.execute(
            "SELECT COUNT(*) AS total, MIN(created_at) AS first_at, MAX(created_at) AS last_at "
            "FROM messages WHERE conversation_id IS NULL"
        ).fetchone()
        if orphan["total"]:
            first_at = orphan["first_at"] or datetime.now().isoformat(timespec="seconds")
            last_at = orphan["last_at"] or first_at
            cursor = connection.execute(
                "INSERT INTO conversations (title, created_at, updated_at) VALUES (?, ?, ?)",
                ("Conversación anterior", first_at, last_at),
            )
            connection.execute(
                "UPDATE messages SET conversation_id = ? WHERE conversation_id IS NULL",
                (cursor.lastrowid,),
            )


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def current_datetime_text() -> str:
    current = datetime.now().astimezone()
    return current.strftime("%A %d/%m/%Y, %H:%M:%S %Z")


def local_tool_answer(user_message: str) -> str | None:
    text = user_message.lower()
    asks_time = any(
        phrase in text
        for phrase in ["que hora", "qué hora", "dime la hora", "hora es", "hora actual"]
    )
    asks_date = any(
        phrase in text
        for phrase in ["que dia", "qué dia", "qué día", "fecha", "dia es hoy", "día es hoy"]
    )
    if asks_time or asks_date:
        return f"La fecha y hora local actual es: {current_datetime_text()}."
    return None


def should_search_web(user_message: str) -> bool:
    text = user_message.lower().strip()
    if re.fullmatch(r"[\s¿?¡!]*(hola|buenas|buenos días|buenos dias|buenas tardes|buenas noches|gracias|ok|vale|perfecto)[\s.!?¿¡]*", text):
        return False

    explicit_search_phrases = [
        "busca", "buscar", "búscame", "buscame", "investiga",
        "consulta en internet", "busca en internet", "busca en la web",
        "en internet", "en la web", "fuentes", "cita fuentes",
    ]
    return requires_current_web(user_message) or any(
        phrase in text for phrase in explicit_search_phrases
    )


def is_physics_question(user_message: str) -> bool:
    normalized = unicodedata.normalize("NFD", user_message.lower())
    text = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    keywords = (
        "fisica", "caida libre", "movimiento rectilineo", "mruv", "aceleracion",
        "velocidad inicial", "gravedad", "energia potencial", "energia cinetica",
        "tiempo de encuentro", "altura inicial", "plano inclinado", "rozamiento",
        "friccion", "fuerza", "masa", "lanzado", "cae", "caer", "colision", "choque",
    )
    return any(keyword in text for keyword in keywords)


def is_math_science_question(user_message: str) -> bool:
    """Route explicit math and science questions to the stronger STEM engine."""
    normalized = unicodedata.normalize("NFD", user_message.lower())
    text = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    keywords = (
        "matematicas", "matematica", "calculo", "calcula", "calcular", "resuelve",
        "ecuacion", "sistema de ecuaciones", "derivada", "integral", "limite", "matriz",
        "geometria", "trigonometria", "probabilidad", "estadistica", "fraccion",
        "fisica", "quimica", "biologia", "ciencias", "cientifico", "cientifica",
        "experimento", "atomo", "molecula", "reaccion quimica", "termodinamica",
        "electricidad", "circuito", "fuerza", "velocidad", "aceleracion", "gravedad",
        "energia", "movimiento", "plano inclinado", "rozamiento", "friccion",
        "masa", "densidad", "presion", "volumen", "newton", "ley de ohm",
    )
    return any(keyword in text for keyword in keywords)


def ask_gemini_stem(user_message: str, physics_mode: bool, on_chunk=None) -> str | None:
    """Use Gemini for STEM; retry its lighter model when the primary is unavailable."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None

    if physics_mode:
        system_prompt = (
            "Eres profesora rigurosa de Fisica y Matematicas de Bachillerato. "
            "Responde en espanol paso a paso. Define datos e incognitas; explica el "
            "modelo fisico, los ejes y las fuerzas; plantea ecuaciones antes de sustituir; "
            "cuida unidades, signos y cifras; verifica el resultado. Si faltan datos, dilo. "
            "No inventes pasos ni resultados."
        )
    else:
        system_prompt = (
            "Eres una profesora rigurosa de matematicas y ciencias de Bachillerato. "
            "Responde en espanol claro y paso a paso. Define las variables, explica el "
            "metodo, muestra los calculos con unidades cuando correspondan y comprueba "
            "el resultado. Distingue hechos de hipotesis; si faltan datos, pregunta."
        )
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_message}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 8192},
    }
    models = (GEMINI_MODEL, GEMINI_FALLBACK_MODEL)
    last_error = None

    for index, model in enumerate(models):
        answer_parts: list[str] = []
        url = (
            GEMINI_API_URL if model == GEMINI_MODEL else
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse"
        )
        try:
            with requests.post(
                url,
                headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                json=payload,
                stream=True,
                timeout=(10, 120),
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines(decode_unicode=False):
                    if isinstance(line, bytes):
                        line = line.decode("utf-8", errors="replace")
                    if not line or not line.startswith("data:"):
                        continue
                    event_data = line[5:].strip()
                    if not event_data or event_data == "[DONE]":
                        continue
                    try:
                        data = json.loads(event_data)
                    except (TypeError, ValueError):
                        continue
                    chunk = "".join(
                        part.get("text", "")
                        for candidate in data.get("candidates", [])[:1]
                        for part in candidate.get("content", {}).get("parts", [])
                        if part.get("text")
                    )
                    if chunk:
                        answer_parts.append(chunk)
                        if on_chunk:
                            on_chunk("".join(answer_parts))

            answer = "".join(answer_parts).strip()
            if answer:
                st.caption(f"Gemini respondio con {model}.")
                return answer
            last_error = "Gemini devolvio una respuesta vacia."
        except requests.exceptions.HTTPError as exc:
            if answer_parts:
                st.caption(f"Gemini respondio parcialmente con {model}.")
                return "".join(answer_parts).strip()
            status = exc.response.status_code if exc.response is not None else "desconocido"
            last_error = f"HTTP {status}"
            retryable = status in (429, 500, 502, 503, 504)
            if not retryable or index == len(models) - 1:
                break
        except requests.exceptions.RequestException as exc:
            if answer_parts:
                st.caption(f"Gemini respondio parcialmente con {model}.")
                return "".join(answer_parts).strip()
            last_error = type(exc).__name__
            if index == len(models) - 1:
                break
        except (ValueError, KeyError, TypeError) as exc:
            if answer_parts:
                st.caption(f"Gemini respondio parcialmente con {model}.")
                return "".join(answer_parts).strip()
            last_error = type(exc).__name__
            if index == len(models) - 1:
                break

        if index < len(models) - 1:
            st.info(f"{model} no esta disponible ({last_error}); probando Gemini de respaldo {models[index + 1]}.")

    st.warning(f"Gemini no pudo responder ({last_error}); Erika continuara con Ollama local.")
    return None

def preferred_physics_model(selected_model: str) -> str:
    """Use the installed reasoning model for physics, falling back to the user's choice."""
    available = list_ollama_models()
    for candidate in ("qwen3:1.7b", "qwen3:4b", "qwen2-math:1.5b-instruct"):
        if candidate in available:
            return candidate
    return selected_model

def requires_current_web(user_message: str) -> bool:
    text = user_message.lower()
    current_phrases = [
        "actual", "actualmente", "ahora", "hoy", "esta semana", "este mes",
        "esta temporada", "último", "última", "ultimos", "últimos",
        "reciente", "noticias", "precio", "cuesta", "cotización", "cotizacion",
        "clima", "temperatura", "resultado", "presidente", "ceo",
        str(datetime.now().year),
    ]
    return any(phrase in text for phrase in current_phrases)


def is_us_president_question(user_message: str) -> bool:
    text = user_message.lower()
    asks_president = "presidente" in text
    asks_us = any(
        phrase in text
        for phrase in [
            "estados unidos",
            "eeuu",
            "ee.uu",
            "usa",
            "united states",
            "estados unidos de america",
            "estados unidos de américa",
        ]
    )
    asks_current = any(phrase in text for phrase in ["actual", "ahora", "hoy", "2025", "2026"])
    return asks_president and asks_us and asks_current


def is_spain_president_question(user_message: str) -> bool:
    text = user_message.lower()
    asks_president = "presidente" in text
    asks_spain = any(phrase in text for phrase in ["españa", "espana", "spain"])
    asks_current = any(phrase in text for phrase in ["actual", "ahora", "hoy", "2025", "2026"])
    return asks_president and asks_spain and asks_current


def is_italy_president_question(user_message: str) -> bool:
    text = user_message.lower()
    asks_president = "presidente" in text
    asks_italy = any(phrase in text for phrase in ["italia", "italy", "italiana", "italiano"])
    asks_current = any(phrase in text for phrase in ["actual", "ahora", "hoy", "2025", "2026"])
    return asks_president and asks_italy and asks_current


def is_direct_current_office_question(user_message: str) -> bool:
    return (
        is_us_president_question(user_message)
        or is_spain_president_question(user_message)
        or is_italy_president_question(user_message)
    )


def direct_us_president_answer(sources: list[WebSource]) -> str | None:
    if not sources:
        return None

    combined = " ".join(f"{source.title} {source.text}" for source in sources).lower()
    has_trump = "donald trump" in combined or "trump" in combined
    has_president = "president" in combined or "presidente" in combined
    has_current_context = any(
        phrase in combined
        for phrase in [
            "47th president",
            "47.º presidente",
            "47 presidente",
            "asume como presidente",
            "assumes office",
            "second term",
            "segundo mandato",
            "casa blanca",
            "white house",
        ]
    )

    if has_trump and has_president and has_current_context:
        return (
            f"El presidente actual de Estados Unidos es **Donald Trump**. "
            f"Fecha de comprobacion local: {current_datetime_text()}."
        )

    return None


def direct_spain_president_answer(sources: list[WebSource]) -> str | None:
    if not sources:
        return None

    combined = " ".join(f"{source.title} {source.text}" for source in sources).lower()
    has_pedro = "pedro sánchez" in combined or "pedro sanchez" in combined
    has_role = (
        "presidente del gobierno" in combined
        or "presidente de españa" in combined
        or "presidente del gobierno de españa" in combined
        or "presidencia del gobierno" in combined
    )

    if has_pedro and has_role:
        return (
            f"El presidente del Gobierno de España actual es **Pedro Sánchez**. "
            f"Fecha de comprobacion local: {current_datetime_text()}."
        )

    return None


def direct_italy_president_answer(sources: list[WebSource]) -> str | None:
    if not sources:
        return None

    combined = " ".join(f"{source.title} {source.text}" for source in sources).lower()
    has_mattarella = "sergio mattarella" in combined or "mattarella" in combined
    has_role = (
        "presidente de la república italiana" in combined
        or "presidente della repubblica" in combined
        or "presidente de italia" in combined
        or "presidente italiano" in combined
    )

    if has_mattarella and has_role:
        return (
            f"El presidente actual de Italia es **Sergio Mattarella**. "
            f"Fecha de comprobacion local: {current_datetime_text()}."
        )

    return None


def direct_current_office_answer(user_message: str, sources: list[WebSource]) -> str | None:
    if is_us_president_question(user_message):
        return direct_us_president_answer(sources)
    if is_spain_president_question(user_message):
        return direct_spain_president_answer(sources)
    if is_italy_president_question(user_message):
        return direct_italy_president_answer(sources)
    return None


def web_query_from_message(user_message: str) -> str:
    """Limpia el query para mejor búsqueda"""
    query = user_message.strip()
    phrases_to_remove = [
        r"\bpor favor\b", r"\bhola\b", r"\bbuenos dias\b", r"\bbuenas tardes\b",
        r"\bme puedes decir\b", r"\bpuedes decirme\b", r"\bdime\b",
        r"\bcuentame\b", r"\bsabes\b", r"\bquiero saber\b",
        r"\bnecesito saber\b", r"\bestoy buscando\b",
    ]
    for pattern in phrases_to_remove:
        query = re.sub(pattern, " ", query, flags=re.IGNORECASE)
    query = re.sub(r"[¿?¡!.,;:]+", " ", query)
    query = re.sub(r"\s+", " ", query).strip()
    return query or user_message


def generate_search_variants(user_message: str) -> list[str]:
    """Genera múltiples variantes del query para mejorar resultados"""
    base = web_query_from_message(user_message)
    variants = [base]

    # Variante 1: agregar "resultado" si es pelea/partido
    lower = base.lower()
    if is_us_president_question(user_message):
        variants = [
            "current president of the United States official",
            "presidente actual de Estados Unidos 2026",
            "president of the United States White House current",
            "Donald Trump president United States 2026",
            base,
        ]
    elif is_spain_president_question(user_message):
        variants = [
            "presidente del Gobierno de España actual Moncloa",
            "Pedro Sánchez presidente del Gobierno España actual",
            "composición del Gobierno España presidente Pedro Sánchez Moncloa",
            "current prime minister Spain Pedro Sanchez",
            base,
        ]
    elif is_italy_president_question(user_message):
        variants = [
            "presidente Repubblica Italiana Sergio Mattarella official",
            "presidente actual de Italia Sergio Mattarella",
            "Quirinale Presidente Sergio Mattarella",
            "current president of Italy Sergio Mattarella",
            base,
        ]

    if any(w in lower for w in ["pelea", "combate", "partido"]):
        if "resultado" not in lower:
            variants.append(f"{base} resultado")

    # Variante 2: agregar "ganador"
    if any(w in lower for w in ["pelea", "combate", "ganó", "gano"]):
        if "ganador" not in lower:
            variants.append(f"{base} ganador")

    # Variante 3: en inglés (a veces encuentra más resultados)
    translations = {
        "pelea": "fight", "ganó": "won", "gano": "won", "como fue": "how was",
        "resultado": "result", "ganador": "winner", "combate": "fight",
        "quien es": "who is", "precio": "price", "clima": "weather",
        "noticias": "news", "hoy": "today", "temperatura": "temperature",
    }
    english = base.lower()
    for es, en in translations.items():
        english = english.replace(es, en)
    if english != base.lower():
        variants.append(english)

    # Variante 4: agregar año actual
    current_year = datetime.now().year
    if str(current_year) not in base:
        variants.append(f"{base} {current_year}")
    if "actualizado" not in lower and "actual" not in lower:
        variants.append(f"{base} actualizado")
    if "noticias" not in lower and any(w in lower for w in ["hoy", "reciente", "ultimo", "último"]):
        variants.append(f"{base} noticias")

    return list(dict.fromkeys(variants))[:6]  # máximo 6 variantes únicas


# ============================================================
# MOTOR 1: TAVILY
# ============================================================
def tavily_search(query: str, limit: int = MAX_WEB_RESULTS) -> list[WebSource]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key or not TAVILY_AVAILABLE:
        raise Exception("Tavily no disponible")
    client = TavilyClient(api_key=api_key)
    response = client.search(
        query=query,
        max_results=limit,
        search_depth="basic",
        timeout=8,
        include_answer=True,
        include_raw_content=False,
    )
    sources = []
    if response.get("answer"):
        sources.append(WebSource(
            title="Respuesta directa",
            url="https://tavily.com",
            text=response["answer"],
            source="Tavily"
        ))
    for result in response.get("results", []):
        sources.append(WebSource(
            title=result.get("title", "Sin titulo"),
            url=result.get("url", ""),
            text=result.get("content", ""),
            source="Tavily"
        ))
    if not sources:
        raise Exception("Tavily: 0 resultados")
    return sources


# ============================================================
# MOTOR 2: SERPAPI
# ============================================================
def serpapi_search(query: str, limit: int = MAX_WEB_RESULTS) -> list[WebSource]:
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key or not SERPAPI_AVAILABLE:
        raise Exception("SerpAPI no disponible")

    params = {
        "q": query,
        "api_key": api_key,
        "num": limit,
        "hl": "es",
        "gl": "es",
    }

    search = GoogleSearch(params, timeout=8)
    results_dict = search.get_dict()

    sources = []
    if "answer_box" in results_dict:
        ab = results_dict["answer_box"]
        answer_text = ab.get("answer") or ab.get("snippet") or ab.get("title", "")
        if answer_text:
            sources.append(WebSource(
                title="Respuesta de Google",
                url=ab.get("link", "https://google.com"),
                text=answer_text,
                source="SerpAPI"
            ))

    if "knowledge_graph" in results_dict:
        kg = results_dict["knowledge_graph"]
        desc = kg.get("description", "")
        if desc:
            sources.append(WebSource(
                title=f"📚 {kg.get('title', 'Knowledge Graph')}",
                url=kg.get("source", {}).get("link", "https://google.com"),
                text=desc,
                source="SerpAPI"
            ))

    for result in results_dict.get("organic_results", []):
        sources.append(WebSource(
            title=result.get("title", "Sin titulo"),
            url=result.get("link", ""),
            text=result.get("snippet", ""),
            source="SerpAPI"
        ))

    if not sources:
        raise Exception("SerpAPI: 0 resultados")
    return sources


# ============================================================
# MOTOR 3-4: DUCKDUCKGO + BING
# ============================================================
def clean_duckduckgo_url(url: str) -> str:
    if url.startswith("//"):
        url = "https:" + url
    parsed = urlparse(url)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path == "/l/":
        uddg = parse_qs(parsed.query).get("uddg", [])
        if uddg:
            return unquote(uddg[0])
    return url


def get_url(url: str, timeout: int = 5) -> requests.Response:
    headers = {"User-Agent": USER_AGENT}
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
    except requests.exceptions.SSLError:
        response = requests.get(url, headers=headers, timeout=timeout, verify=False)
    response.raise_for_status()
    return response


def duckduckgo_search(query: str, limit: int = MAX_WEB_RESULTS) -> list[tuple[str, str]]:
    search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    response = get_url(search_url)
    soup = BeautifulSoup(response.text, "html.parser")
    results = []
    for result in soup.select(".result"):
        title_link = result.select_one(".result__a")
        if not title_link:
            continue
        title = title_link.get_text(" ", strip=True)
        url = clean_duckduckgo_url(title_link.get("href", ""))
        if not title or not url.startswith(("http://", "https://")):
            continue
        results.append((title, url))
        if len(results) >= limit:
            break
    return results


def bing_search(query: str, limit: int = MAX_WEB_RESULTS) -> list[tuple[str, str]]:
    search_url = f"https://www.bing.com/search?q={quote_plus(query)}"
    response = get_url(search_url)
    soup = BeautifulSoup(response.text, "html.parser")
    results = []
    for result in soup.select("li.b_algo"):
        title_link = result.select_one("h2 a")
        if not title_link:
            continue
        title = title_link.get_text(" ", strip=True)
        url = title_link.get("href", "")
        if not title or not url.startswith(("http://", "https://")):
            continue
        results.append((title, url))
        if len(results) >= limit:
            break
    return results


def extract_page_text(url: str) -> str:
    try:
        response = get_url(url, timeout=4)
        content_type = response.headers.get("content-type", "").lower()
        if "text/html" not in content_type:
            return ""
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()
        text = soup.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text)
        return text[:MAX_WEB_CHARS_PER_PAGE]
    except Exception:
        return ""


def fallback_web_search(query: str, limit: int = MAX_WEB_RESULTS) -> list[WebSource]:
    sources: list[WebSource] = []
    for engine_name, search_fn in [
        ("DuckDuckGo", duckduckgo_search),
        ("Bing", bing_search),
    ]:
        try:
            results = search_fn(query, min(limit, 2))
        except Exception:
            continue
        if not results:
            continue
        for title, url in results:
            try:
                text = extract_page_text(url)
            except Exception:
                continue
            if len(text) < 100:
                continue
            sources.append(WebSource(
                title=title, url=url, text=text, source=engine_name
            ))
            if len(sources) >= limit:
                break
        if sources:
            break
    return sources


# ============================================================
# RECOLECTOR INTELIGENTE - PRUEBA MULTIPLES QUERIES
# ============================================================
def gather_web_sources(user_message: str) -> list[WebSource]:
    """Prueba múltiples variantes del query hasta encontrar resultados"""
    sources = []
    variants = generate_search_variants(user_message)[:1]
    errors = []

    # Intentar con cada variante
    for query in variants:
        # Motor 1: Tavily
        if TAVILY_AVAILABLE and os.getenv("TAVILY_API_KEY"):
            try:
                sources = tavily_search(query, MAX_WEB_RESULTS)
                if sources:
                    return sources
            except Exception as exc:
                errors.append(f"Tavily [{query[:30]}]: {str(exc)[:60]}")

        # Motor 2: SerpAPI
        if SERPAPI_AVAILABLE and os.getenv("SERPAPI_API_KEY"):
            try:
                sources = serpapi_search(query, MAX_WEB_RESULTS)
                if sources:
                    return sources
            except Exception as exc:
                errors.append(f"SerpAPI [{query[:30]}]: {str(exc)[:60]}")

    # Fallback: DuckDuckGo + Bing
    for query in variants:
        fallback = fallback_web_search(query, MAX_WEB_RESULTS)
        if fallback:
            return fallback

    if errors:
        st.warning(f"Detalle: {errors[0]}")

    return sources


def format_web_sources(sources: list[WebSource]) -> str:
    if not sources:
        return ""
    blocks = []
    for index, source in enumerate(sources, start=1):
        text = source.text[:1000] if len(source.text) > 1000 else source.text
        blocks.append(f"[Fuente {index}] {source.title}\n{text}")
    return "\n\n".join(blocks)


def sources_markdown(sources: list[WebSource]) -> str:
    if not sources:
        return ""
    lines = ["\n\n---\n**📚 Fuentes consultadas:**"]
    for index, source in enumerate(sources, start=1):
        title = source.title[:60] + "..." if len(source.title) > 60 else source.title
        engine = f" _{source.source}_" if source.source else ""
        lines.append(f"{index}. [{title}]({source.url}){engine}")
    return "\n".join(lines)


def create_conversation(title: str = "Nueva conversación") -> int:
    timestamp = now()
    with db() as connection:
        cursor = connection.execute(
            "INSERT INTO conversations (title, created_at, updated_at) VALUES (?, ?, ?)",
            (title, timestamp, timestamp),
        )
        return int(cursor.lastrowid)


def list_conversations(limit: int = 100) -> list[sqlite3.Row]:
    with db() as connection:
        conversations = connection.execute(
            """SELECT c.id, c.title, c.created_at, c.updated_at,
                      (SELECT content FROM messages m
                       WHERE m.conversation_id = c.id AND m.role = 'Usuario'
                       ORDER BY m.id ASC LIMIT 1) AS preview
               FROM conversations c
               ORDER BY c.updated_at DESC, c.id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        messages_by_conversation: dict[int, list[str]] = {}
        all_messages = connection.execute(
            """SELECT conversation_id, content FROM messages
               WHERE role = 'Usuario' AND conversation_id IN (
                   SELECT id FROM conversations ORDER BY updated_at DESC, id DESC LIMIT ?
               ) ORDER BY id ASC""",
            (limit,),
        ).fetchall()
        for message in all_messages:
            messages_by_conversation.setdefault(int(message["conversation_id"]), []).append(message["content"])

        for conversation in conversations:
            if not conversation["preview"]:
                continue
            messages = messages_by_conversation.get(int(conversation["id"]), [])
            topic_title = conversation_title_from_messages(messages)
            if topic_title != conversation["title"]:
                connection.execute(
                    "UPDATE conversations SET title = ? WHERE id = ?",
                    (topic_title, conversation["id"]),
                )
        return connection.execute(
            """SELECT c.id, c.title, c.created_at, c.updated_at,
                      (SELECT content FROM messages m
                       WHERE m.conversation_id = c.id AND m.role = 'Usuario'
                       ORDER BY m.id ASC LIMIT 1) AS preview
               FROM conversations c
               ORDER BY c.updated_at DESC, c.id DESC LIMIT ?""",
            (limit,),
        ).fetchall()


def delete_conversation(conversation_id: int) -> None:
    with db() as connection:
        connection.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        connection.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))


def conversation_title_from_messages(user_messages: list[str]) -> str:
    """Create a short topic-aware title from the latest recognizable user subject."""
    categories = (
        ("Literatura", ("literatura", "poema", "poesia", "novela", "poeta", "metafora", "verso", "teatro", "comentario de texto", "figura literaria", "lorca", "cervantes")),
        ("Fisica", ("fisica", "plano inclinado", "fuerza", "aceleracion", "gravedad", "velocidad", "energia", "movimiento", "rozamiento", "friccion", "newton", "ohm")),
        ("Quimica", ("quimica", "atomo", "molecula", "reaccion", "enlace quimico", "ph", "estequiometria")),
        ("Biologia", ("biologia", "celula", "adn", "genetica", "ecosistema", "mitosis", "organismo")),
        ("Matematicas", ("matematicas", "matematica", "ecuacion", "integral", "derivada", "calculo", "geometria", "probabilidad", "fraccion", "funcion", "teorema")),
        ("Historia", ("historia", "guerra", "revolucion", "imperio", "siglo", "edad media", "segunda guerra")),
        ("Geografia", ("geografia", "clima", "continente", "rio", "poblacion", "mapa", "relieve")),
        ("Programacion", ("programacion", "python", "javascript", "codigo", "bug", "error de codigo", "programa", "funcion en")),
        ("Ingles", ("ingles", "english", "traduce", "traducir", "grammar", "vocabulario")),
        ("Ciencias", ("ciencia", "ciencias", "experimento", "metodo cientifico")),
        ("Filosofia", ("filosofia", "platon", "aristoteles", "etica", "existencialismo")),
        ("Economia", ("economia", "inflacion", "oferta y demanda", "pib", "mercado")),
    )

    chosen_message = user_messages[0] if user_messages else "Nueva conversacion"
    chosen_subject = None
    for message in reversed(user_messages):
        normalized = unicodedata.normalize("NFD", message.casefold())
        normalized = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
        match = next(
            (label for label, keywords in categories if any(keyword in normalized for keyword in keywords)),
            None,
        )
        if match:
            chosen_subject = match
            chosen_message = message
            break

    snippet = re.sub(r"\s+", " ", chosen_message).strip(" \t\r\n???!.,:;-")
    snippet = re.sub(
        r"^(hola|oye|por favor|ayudame|puedes ayudarme|explicame|resuelve|calcula|analiza|comenta)\s+",
        "", snippet, flags=re.IGNORECASE,
    )
    snippet = snippet[:42].rstrip(" .,;:-")
    if snippet:
        snippet = snippet[0].upper() + snippet[1:]
    if chosen_subject and snippet:
        labels = {"Fisica": "F\u00edsica", "Quimica": "Qu\u00edmica", "Matematicas": "Matem\u00e1ticas", "Biologia": "Biolog\u00eda", "Ingles": "Ingl\u00e9s", "Geografia": "Geograf\u00eda", "Filosofia": "Filosof\u00eda"}
        return f"{labels.get(chosen_subject, chosen_subject)} \u00b7 {snippet}"[:64]
    if chosen_subject:
        labels = {"Fisica": "F\u00edsica", "Quimica": "Qu\u00edmica", "Matematicas": "Matem\u00e1ticas", "Biologia": "Biolog\u00eda", "Ingles": "Ingl\u00e9s", "Geografia": "Geograf\u00eda", "Filosofia": "Filosof\u00eda"}
        return labels.get(chosen_subject, chosen_subject)
    return snippet[:64] or "Nueva conversacion"


def welcome_phrase(conversation_id: int) -> str:
    """Keep one varied welcome line for each new conversation in this session."""
    key = f"welcome_phrase_{conversation_id}"
    if key not in st.session_state:
        phrases = [
            "¿Por dónde empezamos?",
            "¿Qué te gustaría descubrir hoy?",
            "Cuéntame qué tienes en mente.",
            "¿Qué idea, duda o reto traes hoy?",
            "Vamos a darle forma a esa idea.",
            "¿En qué te puedo ayudar hoy?",
            "¿Qué te gustaría resolver o aprender?",
        ]
        previous_phrase = st.session_state.get("last_welcome_phrase")
        choices = [item for item in phrases if item != previous_phrase]
        st.session_state[key] = random.choice(choices)
        st.session_state["last_welcome_phrase"] = st.session_state[key]
    return st.session_state[key]


def save_message(
    role: str, content: str, model: str | None = None,
    conversation_id: int | None = None,
) -> int:
    if conversation_id is None:
        conversation_id = st.session_state.get("active_conversation_id")
    if conversation_id is None:
        conversation_id = create_conversation()
        st.session_state["active_conversation_id"] = conversation_id

    timestamp = now()
    with db() as connection:
        cursor = connection.execute(
            """INSERT INTO messages
               (role, content, model, created_at, conversation_id)
               VALUES (?, ?, ?, ?, ?)""",
            (role, content, model, timestamp, conversation_id),
        )
        if role == "Usuario":
            user_messages = [
                row[0] for row in connection.execute(
                    "SELECT content FROM messages WHERE conversation_id = ? AND role = 'Usuario' ORDER BY id ASC",
                    (conversation_id,),
                ).fetchall()
            ]
            title = conversation_title_from_messages(user_messages)
            connection.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title, timestamp, conversation_id),
            )
        else:
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (timestamp, conversation_id),
            )
        return int(cursor.lastrowid)


def save_memory(content: str, source_message_id: int | None = None) -> None:
    clean_content = content.strip()
    if not clean_content:
        return
    with db() as connection:
        connection.execute(
            "INSERT OR IGNORE INTO memories (content, source_message_id, created_at) VALUES (?, ?, ?)",
            (clean_content, source_message_id, now()),
        )


def load_recent_messages(
    limit: int = MAX_CONTEXT_MESSAGES, conversation_id: int | None = None,
) -> list[sqlite3.Row]:
    if conversation_id is None:
        conversation_id = st.session_state.get("active_conversation_id")
    with db() as connection:
        if conversation_id is None:
            rows = connection.execute(
                "SELECT role, content FROM messages ORDER BY id DESC LIMIT ?", (limit,),
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
                (conversation_id, limit),
            ).fetchall()
    return list(reversed(rows))


def load_chat_history(conversation_id: int | None = None) -> list[sqlite3.Row]:
    if conversation_id is None:
        conversation_id = st.session_state.get("active_conversation_id")
    if conversation_id is None:
        return []
    with db() as connection:
        return connection.execute(
            "SELECT role, content, created_at FROM messages WHERE conversation_id = ? ORDER BY id ASC",
            (conversation_id,),
        ).fetchall()


def load_memory_count() -> int:
    with db() as connection:
        row = connection.execute("SELECT COUNT(*) AS total FROM memories").fetchone()
    return int(row["total"])


def clear_all_memory() -> None:
    with db() as connection:
        connection.execute("DELETE FROM messages")
        connection.execute("DELETE FROM memories")
        connection.execute("DELETE FROM conversations")
    st.session_state.pop("active_conversation_id", None)
    st.session_state.pop("pending_delete_conversation", None)


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z0-9áéíóúÁÉÍÓÚñÑüÜ]+", text.lower())
    stop_words = {
        "algo", "como", "con", "cuando", "de", "del", "el", "en", "es", "esa",
        "ese", "esto", "la", "las", "lo", "los", "me", "mi", "para", "por",
        "que", "se", "si", "sin", "tu", "un", "una", "y", "yo",
    }
    return [word for word in words if len(word) > 2 and word not in stop_words]


def find_relevant_memories(prompt: str, limit: int = MAX_MEMORIES) -> list[str]:
    query_words = Counter(tokenize(prompt))
    if not query_words:
        return []
    with db() as connection:
        rows = connection.execute(
            "SELECT content FROM memories ORDER BY id DESC LIMIT 100"
        ).fetchall()
    scored_memories = []
    for row in rows:
        memory = row["content"]
        memory_words = Counter(tokenize(memory))
        score = sum(query_words[word] * memory_words[word] for word in query_words)
        if score:
            scored_memories.append((score, memory))
    scored_memories.sort(key=lambda item: item[0], reverse=True)
    return [memory for _, memory in scored_memories[:limit]]


def auto_learn_from_user_message(message: str, message_id: int) -> None:
    cleaned = message.strip()
    if len(cleaned) < 12:
        return
    learning_patterns = [
        r"\bme llamo\b", r"\bmi nombre es\b", r"\bsoy\b", r"\bme gusta\b",
        r"\bno me gusta\b", r"\bprefiero\b", r"\btrabajo\b", r"\bquiero\b",
        r"\brecuerda\b", r"\baprende\b",
    ]
    if any(re.search(pattern, cleaned.lower()) for pattern in learning_patterns):
        save_memory(cleaned, message_id)


def list_ollama_models() -> list[str]:
    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=5)
        response.raise_for_status()
        data = response.json()
    except Exception:
        return DEFAULT_MODELS
    EXCLUDE_KEYWORDS = ["embed", "embedding", "nomic-embed", "mxbai", "all-minilm"]
    models = [
        item["name"] for item in data.get("models", [])
        if "name" in item
        and not any(kw in item["name"].lower() for kw in EXCLUDE_KEYWORDS)
    ]
    return models or DEFAULT_MODELS


def build_prompt(
    user_message: str, web_sources: list[WebSource] | None = None
) -> tuple[str, str]:
    web_sources = web_sources or []

    if web_sources:
        system_instruction = (
            "Eres Erika, una IA local con acceso a una herramienta de busqueda web. "
            f"Fecha y hora local actual: {current_datetime_text()}. "
            f"{MODEL_KNOWLEDGE_NOTE} "
            "Responde usando principalmente DATOS DE INTERNET. "
            "Si las fuentes no contienen un dato, dilo claramente. "
            "No digas que tu conocimiento llega hasta 2023 si se te han dado fuentes web. "
            "No digas que no tienes internet si se te han dado fuentes web. "
            "Cita fuentes al final. Responde en Espanol de forma clara y concisa."
        )
    else:
        system_instruction = (
            "Eres Erika, una IA local. "
            f"Fecha y hora local actual: {current_datetime_text()}. "
            f"{MODEL_KNOWLEDGE_NOTE} "
            "Ahora no se han proporcionado fuentes web para esta respuesta. "
            "Si la pregunta necesita datos actuales, indica que debe activarse la busqueda web. "
            "Responde en Espanol."
        )

    if web_sources:
        web_block = format_web_sources(web_sources)
        user_content = f"""DATOS DE INTERNET (informacion verificada):

{web_block}

PREGUNTA: {user_message}

Responde con datos actuales de las fuentes anteriores. Si hay conflicto, dilo."""
    else:
        memories = find_relevant_memories(user_message)
        memory_block = "\n".join(f"- {memory}" for memory in memories)
        user_content = f"""Memoria relevante:
{memory_block or "- No hay memoria relevante."}

Pregunta:
{user_message}"""

    return system_instruction, user_content


def recent_conversation_messages(
    user_message: str, conversation_id: int | None = None,
) -> list[dict[str, str]]:
    rows = load_recent_messages(MAX_CONTEXT_MESSAGES + 1, conversation_id)
    if rows and rows[-1]["role"] == "Usuario" and rows[-1]["content"] == user_message:
        rows = rows[:-1]
    return [
        {
            "role": "user" if row["role"] == "Usuario" else "assistant",
            "content": row["content"],
        }
        for row in rows
        if row["role"] in {"Usuario", "IA"} and row["content"].strip()
    ]


def ask_ollama(
    model: str,
    user_message: str,
    allow_web: bool,
    rag_enabled: bool = True,
    conversation_id: int | None = None,
    on_chunk=None,
    local_only: bool = False,
) -> str | None:
    if any(keyword in model.lower() for keyword in ["embed", "embedding"]):
        st.error(f"❌ El modelo '{model}' no soporta chat.")
        return None

    conversation_history = recent_conversation_messages(user_message, conversation_id)
    web_sources: list[WebSource] = []
    must_use_web = False if local_only else requires_current_web(user_message)
    should_use_web = False if local_only else should_search_web(user_message)

    if should_use_web and (allow_web or must_use_web):
        if must_use_web and not allow_web:
            st.info("🔍 Esta pregunta necesita datos actuales, asi que usare internet igualmente.")

        with st.status("🔍 Buscando en internet...", expanded=True) as status:
            try:
                variants = generate_search_variants(user_message)
                st.write(f"**Probando {len(variants)} variantes:**")
                for i, v in enumerate(variants, 1):
                    st.write(f"  {i}. `{v}`")

                web_sources = gather_web_sources(user_message)

                if web_sources:
                    engines = list(set(s.source for s in web_sources if s.source))
                    status.update(
                        label=f"✅ {len(web_sources)} fuentes ({', '.join(engines)})",
                        state="complete"
                    )
                    with st.expander(f"📄 Ver {len(web_sources)} fuentes"):
                        for i, s in enumerate(web_sources, 1):
                            st.write(f"**[{i}] {s.title}** _{s.source}_")
                            st.caption(s.text[:400] + "...")
                            st.caption(f"🔗 {s.url}")
                else:
                    status.update(label="❌ Sin resultados en ningún motor", state="error")
            except Exception as exc:
                status.update(label=f"❌ Error: {exc}", state="error")

    if is_direct_current_office_question(user_message):
        direct_answer = direct_current_office_answer(user_message, web_sources)
        if direct_answer:
            return direct_answer + sources_markdown(web_sources)


    if must_use_web and not web_sources:
        return (
            "No puedo responder con seguridad usando solo la memoria interna del modelo, "
            "porque esta pregunta necesita informacion actual. No he conseguido fuentes web "
            "en este intento; revisa la conexion o prueba de nuevo."
        )

    stem_mode = is_math_science_question(user_message)
    physics_mode = is_physics_question(user_message)
    if stem_mode and not local_only and os.getenv("GEMINI_API_KEY", "").strip():
        st.info(f"Intentando Google Gemini ({GEMINI_MODEL}) para esta pregunta de ciencias o matematicas.")
        gemini_answer = ask_gemini_stem(user_message, physics_mode, on_chunk=on_chunk)
        if gemini_answer:
            st.success("Respuesta generada con Google Gemini.")
            return gemini_answer + sources_markdown(web_sources)
        st.warning("Gemini no pudo responder. Erika continuara con los motores locales.")
    elif stem_mode and not local_only:
        st.info("Configura GEMINI_API_KEY en .env para usar Gemini en matematicas y ciencias. Usare Ollama local por ahora.")

    tool_answer = local_tool_answer(user_message)
    if tool_answer:
        return tool_answer

    inclined_plane_answer = physics_response_utils.solve_inclined_plane_braking(user_message)
    if inclined_plane_answer:
        return inclined_plane_answer

    if physics_mode:
        model = preferred_physics_model(model)
        st.caption(f"Motor de f\u00edsica: {model}")
        system_prompt = (
            "Eres profesor de fisica de Bachillerato y resuelves con rigor. "
            "Responde en espanol claro. Antes de calcular, identifica los datos, "
            "las incognitas, las hipotesis y los ejes con sentido positivo. "
            "Describe las fuerzas y su direccion. En planos inclinados, descompone "
            "el peso en mg sin(theta) paralelo y mg cos(theta) perpendicular; calcula "
            "la normal desde el eje perpendicular y el rozamiento como mu*N, opuesto "
            "al movimiento relativo. Aplica Newton por ejes y luego la ecuacion "
            "cinematica adecuada. Si se detiene, fija velocidad final a cero antes "
            "de sustituir. No inventes valores ni velocidades intermedias. Mantiene "
            "unidades, signos y precision hasta el final. Comprueba el resultado "
            "con otra ecuacion o sustitucion. Si faltan datos esenciales, pregunta. "
            "Se conciso: incluye solo el metodo, las ecuaciones necesarias, los resultados y una comprobacion."
        )
        user_prompt = (
            f"Enunciado:\n{user_message}\n\n"
            "Resuelve en menos de 160 palabras: datos y fuerzas solo si hacen falta; "
            "ecuacion, sustitucion numerica, resultados con unidades y una comprobacion."
        )
    else:
        system_prompt, user_prompt = build_prompt(user_message, web_sources)

    if rag_enabled and not stem_mode and not web_sources:
        try:
            rag_answer = rag.ask_with_context(
                user_message, model=model, history=conversation_history, on_chunk=on_chunk,
            )
            if rag_answer and "Error contacting local model" not in rag_answer and rag_answer.strip():
                return rag_answer
        except Exception:
            pass

    generated_parts: list[str] = []
    generation_done_reason = None
    try:
        with st.spinner(f"{model} generando respuesta..."):
            request_messages = [
                {"role": "system", "content": system_prompt},
                *conversation_history,
                {"role": "user", "content": user_prompt},
            ]
            for generation_round in range(3):
                response = requests.post(
                    OLLAMA_CHAT_URL,
                    json={
                        "model": model,
                        "messages": request_messages,
                        "stream": True,
                        **({"think": False} if physics_mode and model.startswith("qwen3") else {}),
                        "keep_alive": "10m",
                        "options": {
                            "temperature": 0.0,
                            "num_ctx": 4096 if physics_mode else 3072,
                            "num_predict": 1024 if physics_mode else 768,
                        },
                    },
                    stream=True,
                    timeout=(5, 90 if physics_mode else 35),
                )
                response.raise_for_status()
                generation_done_reason = None
                for line in response.iter_lines(decode_unicode=False):
                    if not line:
                        continue
                    if isinstance(line, bytes):
                        line = line.decode("utf-8", errors="replace")
                    try:
                        chunk_data = json.loads(line)
                    except (TypeError, ValueError):
                        continue
                    chunk = chunk_data.get("message", {}).get("content", "")
                    if chunk:
                        generated_parts.append(chunk)
                    if chunk_data.get("done"):
                        generation_done_reason = chunk_data.get("done_reason")
                    if on_chunk and (len(generated_parts) % 4 == 0 or chunk_data.get("done")):
                        on_chunk("".join(generated_parts))

                if generation_done_reason != "length" or generation_round == 2:
                    break

                request_messages = [
                    {"role": "system", "content": system_prompt},
                    *conversation_history,
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": "".join(generated_parts)},
                    {
                        "role": "user",
                        "content": "Continua exactamente desde donde se corto la respuesta anterior. "
                                   "No repitas nada y termina todos los apartados pendientes.",
                    },
                ]
    except requests.exceptions.ConnectionError:
        st.error("Ollama no responde en http://127.0.0.1:11434. Cierra Erika y vuelve a abrirla con run_app.bat.")
        return None
    except requests.exceptions.Timeout:
        partial_answer = "".join(generated_parts).strip()
        if partial_answer:
            partial_answer += "\n\n**Aviso:** La generacion se interrumpio por tiempo de espera; esta respuesta puede estar incompleta."
            return partial_answer + sources_markdown(web_sources)
        st.error("Ollama tardó demasiado en empezar. Prueba una pregunta más corta.")
        return None
    except requests.exceptions.HTTPError:
        st.error(f"❌ Error HTTP {response.status_code}")
        try:
            st.code(json.dumps(response.json(), indent=2, ensure_ascii=False))
        except Exception:
            st.code(response.text)
        return None
    except Exception as exc:
        st.error(f"❌ Error: {exc}")
        return None

    answer = "".join(generated_parts).strip()
    if not answer:
        st.error("Ollama no devolvió texto. Inténtalo de nuevo.")
        return None
    if generation_done_reason == "length":
        answer += "\n\n**Aviso:** Erika alcanzo el limite de continuaciones; la respuesta puede estar incompleta."
    return answer + sources_markdown(web_sources)


def export_memory() -> str:
    with db() as connection:
        messages = connection.execute(
            "SELECT role, content, model, created_at, conversation_id FROM messages ORDER BY id ASC"
        ).fetchall()
        conversations = connection.execute(
            "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
        ).fetchall()
        memories = connection.execute(
            "SELECT content, created_at FROM memories ORDER BY id ASC"
        ).fetchall()
    return json.dumps(
        {
            "messages": [dict(row) for row in messages],
            "conversations": [dict(row) for row in conversations],
            "memories": [dict(row) for row in memories],
        },
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# INTERFAZ
# ============================================================
init_db()

st.set_page_config(page_title="Erika - IA local", page_icon=str(Path(__file__).parent / "assets" / "erika_avatar.png"), layout="wide")

st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp, [data-testid="stAppViewContainer"] {
  background: #0b0b0b;
  color: #f3f3f3;
}
[data-testid="stHeader"] { background: rgba(11, 11, 11, .92); }
[data-testid="stToolbar"] { right: 1rem; }
[data-testid="stMainBlockContainer"] { max-width: 1040px; padding: 2.2rem 2rem 7.5rem; }
[data-testid="stSidebar"] { background: #111111; border-right: 1px solid rgba(255,255,255,.07); }
[data-testid="stSidebar"] > div:first-child { background: transparent; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label { color: #c2cfca; }
[data-testid="stSidebar"] [data-testid="stMetricValue"] { color: #f3f8f5; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .78rem; }
[data-testid="stSidebar"] hr { border-color: rgba(180, 210, 200, .12); margin: 1.15rem 0; }
[data-testid="stSidebar"] button {
  border: 1px solid rgba(180, 210, 200, .14); border-radius: 12px;
  color: #e7efeb; background: rgba(255,255,255,.035); transition: .18s ease;
}
[data-testid="stSidebar"] button:hover { border-color: #3ba98d; background: rgba(43, 145, 119, .12); color: #fff; }
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] textarea {
  background: #19211f; border-color: rgba(180, 210, 200, .15); color: #eff5f2; border-radius: 12px;
}
[data-testid="stSidebar"] [data-testid="stMetric"] {
  background: rgba(255,255,255,.035); border: 1px solid rgba(180,210,200,.08);
  border-radius: 14px; padding: .75rem .9rem;
}
.erika-brand { display:flex; align-items:center; gap:12px; padding:.35rem .15rem 1.05rem; }
.erika-mark {
  width:42px; height:42px; display:grid; place-items:center; border-radius:15px;
  color:#071713; font-size:22px; font-weight:700;
  background:linear-gradient(145deg,#79ddbd,#36aa8b); box-shadow:0 5px 22px rgba(49,180,144,.24);
}
.erika-brand-name { color:#f2f7f4; font-weight:700; letter-spacing:.13em; font-size:.94rem; }
.erika-brand-caption { color:#85968f; font-size:.66rem; letter-spacing:.12em; margin-top:2px; }
.st-key-erika_welcome { padding-top:12vh; }
.st-key-erika_logo [data-testid="stImage"] { display:flex; justify-content:center; }
.st-key-erika_sidebar_avatar img { width:46px; height:46px; object-fit:cover; border-radius:14px; border:1px solid rgba(121,221,189,.35); }
.erika-hero { text-align:center; padding:1rem 1rem 1.5rem; }
.erika-eyebrow {
  color:#8bd4ba; font-size:.72rem; letter-spacing:.2em; text-transform:uppercase;
  margin-bottom:1rem; font-weight:650;
}
.erika-hero h1 {
  color:#f3f3f3; font-size:clamp(1.8rem,3vw,2.35rem); line-height:1.2;
  font-weight:550; letter-spacing:-.035em; margin:0 auto .7rem;
}
.erika-hero p { color:#a0a0a0; font-size:.98rem; margin:0; }
.erika-prompt-label { color:#9aa9a3; text-align:center; font-size:.78rem; margin:.9rem 0 .5rem; }
[data-testid="stChatMessage"] {
  max-width: 850px; margin: .65rem auto; padding: 1rem 1.2rem;
  border: 1px solid rgba(180,210,200,.07); border-radius: 18px;
  background: rgba(255,255,255,.025);
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] { color:#e6eeea; line-height:1.75; }
[data-testid="stChatMessage"] [data-testid="stChatMessageAvatar"] {
  background:#1c302b; color:#a2e4ce; border:1px solid rgba(95,190,158,.2);
}
[data-testid="stChatInput"] { max-width: 850px; margin-left:auto; margin-right:auto; }
.erika-suggestion-label { color:#8e8e8e; font-size:.78rem; text-align:center; margin:.4rem 0 .5rem; }
.st-key-erika_suggestions button { min-height:3.2rem; border-radius:16px; border:1px solid #303030; background:#141414; color:#d4d4d4; }
.st-key-erika_suggestions button:hover { background:#202020; border-color:#515151; color:white; }
[data-testid="stChatInput"] > div {
  background:#19211f; border:1px solid rgba(180,210,200,.17); border-radius:20px;
  box-shadow:0 12px 45px rgba(0,0,0,.2);
}
[data-testid="stChatInput"] textarea { color:#f0f5f2; padding:.9rem 1rem; }
[data-testid="stChatInput"] textarea:focus { box-shadow:0 0 0 1px #40b494; }
[data-testid="stChatInput"] button { color:#061510; background:#72d5b5; border-radius:12px; }
[data-testid="stChatInput"] button:hover { background:#94e6cc; }
[data-testid="stAlert"] { border-radius:14px; }
[data-testid="stExpander"] { border-color:rgba(180,210,200,.14); border-radius:12px; background:rgba(255,255,255,.02); }
.stCaption { color:#91a099; }
@media (max-width:700px) {
  [data-testid="stMainBlockContainer"] { padding:1.1rem 1rem 7rem; }
  .st-key-erika_welcome { padding-top:8vh; }
  .erika-hero { padding:1rem .25rem 1.5rem; }
  [data-testid="stChatMessage"] { padding:.8rem .7rem; }
}
</style>
""", unsafe_allow_html=True)

if "active_conversation_id" not in st.session_state:
    existing_conversations = list_conversations()
    st.session_state["active_conversation_id"] = (
        int(existing_conversations[0]["id"]) if existing_conversations else create_conversation()
    )

with st.sidebar:
    brand_columns = st.columns([1, 4])
    with brand_columns[0]:
        with st.container(key="erika_sidebar_avatar"):
            st.image(str(Path(__file__).parent / "assets" / "erika_avatar.png"), width=46)
    with brand_columns[1]:
        st.markdown(
            '<div class="erika-brand-name">ERIKA</div><div class="erika-brand-caption">ASISTENTE PERSONAL</div>',
            unsafe_allow_html=True,
        )
    if st.button("＋  Nueva conversación", key="new_conversation", use_container_width=True):
        st.session_state["active_conversation_id"] = create_conversation()
        st.session_state.pop("pending_delete_conversation", None)
        st.rerun()

    search_chats = st.text_input(
        "Buscar chats", key="conversation_search", placeholder="⌕  Buscar chats…",
        label_visibility="collapsed",
    )
    conversations = list_conversations()
    if search_chats.strip():
        needle = search_chats.strip().casefold()
        conversations = [
            chat for chat in conversations
            if needle in (chat["title"] or "").casefold()
            or needle in (chat["preview"] or "").casefold()
        ]

    if conversations:
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)
        grouped = {"Hoy": [], "Ayer": [], "Anteriores": []}
        for chat in conversations:
            try:
                chat_date = datetime.fromisoformat(chat["updated_at"]).date()
            except (TypeError, ValueError):
                chat_date = today
            label = "Hoy" if chat_date == today else "Ayer" if chat_date == yesterday else "Anteriores"
            grouped[label].append(chat)

        for label, chats in grouped.items():
            if chats:
                st.caption(label)
            for chat in chats:
                title = (chat["title"] or "Nueva conversación").replace("\n", " ").strip()
                if len(title) > 34:
                    title = title[:33].rstrip() + "…"
                chat_cols = st.columns([5, 1])
                with chat_cols[0]:
                    if st.button(
                        title or "Nueva conversación", key=f"open_chat_{chat['id']}",
                        use_container_width=True,
                        type="primary" if int(chat["id"]) == int(st.session_state["active_conversation_id"]) else "secondary",
                    ):
                        st.session_state["active_conversation_id"] = int(chat["id"])
                        st.session_state.pop("pending_delete_conversation", None)
                        st.rerun()
                with chat_cols[1]:
                    if st.button("⋯", key=f"delete_menu_{chat['id']}", help="Opciones del chat"):
                        st.session_state["pending_delete_conversation"] = int(chat["id"])
                        st.rerun()
    elif search_chats.strip():
        st.caption("No hay chats que coincidan con la búsqueda.")

    pending_delete = st.session_state.get("pending_delete_conversation")
    if pending_delete is not None:
        st.warning("¿Eliminar este chat? Esta acción no se puede deshacer.")
        confirm_cols = st.columns(2)
        with confirm_cols[0]:
            if st.button("Eliminar", key="confirm_delete_chat", type="primary", use_container_width=True):
                delete_conversation(int(pending_delete))
                if int(st.session_state["active_conversation_id"]) == int(pending_delete):
                    st.session_state["active_conversation_id"] = create_conversation()
                st.session_state.pop("pending_delete_conversation", None)
                st.rerun()
        with confirm_cols[1]:
            if st.button("Cancelar", key="cancel_delete_chat", use_container_width=True):
                st.session_state.pop("pending_delete_conversation", None)
                st.rerun()

    st.divider()
    st.subheader("Ajustes")
    st.caption("Tu espacio para pensar, aprender y crear.")

    tavily_ok = TAVILY_AVAILABLE and os.getenv("TAVILY_API_KEY")
    serpapi_ok = SERPAPI_AVAILABLE and os.getenv("SERPAPI_API_KEY")
    if tavily_ok:
        st.success("Tavily conectado")
    elif TAVILY_AVAILABLE:
        st.caption("Tavily está instalado; falta configurar su clave.")
    if serpapi_ok:
        st.success("SerpAPI conectado")
    elif SERPAPI_AVAILABLE:
        st.caption("SerpAPI está instalado; falta configurar su clave.")

    st.subheader("Modelos")
    models = list_ollama_models()
    selected_model = st.selectbox("Ollama local", models, index=0)
    gemini_configured = bool(os.getenv("GEMINI_API_KEY", "").strip())
    with st.expander("Gemini \u00b7 Matem\u00e1ticas y ciencias", expanded=True):
        if gemini_configured:
            st.success("Clave configurada \u00b7 selecci\u00f3n autom\u00e1tica")
            st.caption(
                "Gemini responde automaticamente a preguntas de matematicas, fisica, "
                "quimica, biologia y ciencias. Si Gemini 3.8 esta saturado, Erika prueba "
                "Gemini 3.5 Flash-Lite como respaldo."
            )
            st.caption(
                f"El resto de preguntas usa Ollama ({selected_model}). "
                "Si Gemini falla, Erika intenta responder con Ollama."
            )
        else:
            st.warning("Gemini no est\u00e1 configurado")
            st.caption(
                "Sin GEMINI_API_KEY, todas las respuestas se generan con Ollama local. "
                "A\u00f1ade la clave al archivo .env para activar el enrutamiento autom\u00e1tico."
            )
    allow_web = st.toggle("Buscar en internet", value=True)
    rag_enabled = st.toggle("Consultar documentos locales", value=True)
    st.metric("Recuerdos guardados", load_memory_count())

    with st.expander("Crear imagen (local)", expanded=False):
        st.caption(
            "La imagen se genera en este PC con un modelo local. "
            "Puede tardar varios minutos porque este equipo no tiene CUDA disponible."
        )
        with st.form("local_image_generation_form"):
            image_prompt = st.text_area(
                "Describe la imagen",
                placeholder="Una biblioteca fantastica bajo un cielo estrellado...",
                max_chars=600,
                key="local_image_prompt",
            )
            create_image = st.form_submit_button(
                "Generar imagen", use_container_width=True, type="primary"
            )

        if create_image:
            if not image_prompt.strip():
                st.warning("Describe la imagen que quieres crear.")
            else:
                try:
                    with st.spinner("Erika esta creando la imagen en este PC..."):
                        if not local_image_generator.model_is_downloaded():
                            local_image_generator.download_model()
                        generated_path = local_image_generator.generate_image(
                            image_prompt.strip()
                        )
                    st.session_state["erika_last_generated_image"] = str(generated_path)
                    st.success("Imagen creada localmente.")
                except Exception as exc:
                    st.error(f"No se pudo crear la imagen: {exc}")

    with st.expander("Guardar un recuerdo"):
        explicit_memory = st.text_area(
            "Recuerdo", label_visibility="collapsed",
            placeholder="Escribe algo que Erika deba recordar",
        )
        if st.button("Guardar recuerdo", use_container_width=True) and explicit_memory.strip():
            save_memory(explicit_memory)
            st.rerun()

    st.download_button(
        "Exportar memoria", data=export_memory(), file_name="erika_memory.json",
        mime="application/json", use_container_width=True,
    )
    if st.button("Borrar conversaciones y recuerdos", use_container_width=True):
        clear_all_memory()
        st.rerun()


active_conversation_id = int(st.session_state["active_conversation_id"])
history = load_chat_history(active_conversation_id)
suggested_message = None
if not history:
    phrase = welcome_phrase(active_conversation_id)
    with st.container(key="erika_welcome"):
        with st.container(key="erika_logo"):
            st.image(str(Path(__file__).parent / "assets" / "erika_logo.png"), width=176)
        st.markdown(f"""
        <div class="erika-hero">
          <div class="erika-eyebrow">ERIKA / TU ASISTENTE</div>
          <h1>{phrase}</h1>
          <p>Pregunta, estudia, crea o resuelve algo. Empezamos por donde quieras.</p>
        </div>
        """, unsafe_allow_html=True)
    with st.container(key="erika_suggestions"):
        st.markdown('<div class="erika-suggestion-label">O empieza con una de estas ideas</div>', unsafe_allow_html=True)
        prompts = [
            "Ayudame a estudiar un tema",
            "Resuelve un ejercicio de ciencias",
            "Analiza una obra literaria",
        ]
        cols = st.columns(3)
        for index, (col, prompt) in enumerate(zip(cols, prompts)):
            with col:
                if st.button(prompt, key=f"suggest_{active_conversation_id}_{index}", use_container_width=True):
                    suggested_message = prompt

for message in history:
    role = "user" if message["role"] == "Usuario" else "assistant"
    with st.chat_message(role):
        st.markdown(message["content"])

last_generated_image = st.session_state.get("erika_last_generated_image")
if last_generated_image and Path(last_generated_image).is_file():
    with st.container():
        st.markdown("### Imagen creada localmente")
        st.image(last_generated_image, width=512)
        with open(last_generated_image, "rb") as image_file:
            st.download_button(
                "Descargar imagen",
                data=image_file.read(),
                file_name=Path(last_generated_image).name,
                mime="image/png",
                key="download_erika_image",
            )

chat_submission = st.chat_input(
    "Escribe un mensaje a Erika...",
    accept_file=True,
    file_type=["png", "jpg", "jpeg", "webp", "bmp", "pdf", "docx", "txt", "md", "csv"],
    max_upload_size=15,
    key="erika_chat_input",
)
uploaded_file = None
if suggested_message is not None:
    user_message = suggested_message
elif chat_submission is not None:
    user_message = (chat_submission.text or "").strip()
    uploaded_files = chat_submission.files
    uploaded_file = uploaded_files[0] if uploaded_files else None
    if uploaded_file and not user_message:
        user_message = "Analiza el archivo adjunto y dame un resumen con comentarios y sugerencias."
else:
    user_message = None

attachment = None
if user_message and uploaded_file:
    try:
        attachment = local_attachments.extract_attachment(
            uploaded_file.name, uploaded_file.getvalue()
        )
    except (ValueError, Exception) as upload_error:
        st.error(f"No pude leer el archivo: {upload_error}")
        user_message = None

if user_message:
    user_display_message = user_message
    model_message = user_message
    if attachment:
        user_display_message = f"[Archivo adjunto: {attachment['name']}]\\n\\n{user_message}"
        if attachment["kind"] == "document":
            model_message = (
                f"El usuario adjunta el archivo {attachment['name']} y pregunta: "
                f"{user_message}\\n\\nContenido extraido del archivo:\\n"
                f"{attachment['text']}"
            )
    user_message_id = save_message(
        "Usuario", user_display_message, selected_model, active_conversation_id,
    )
    auto_learn_from_user_message(user_message, user_message_id)

    with st.chat_message("user"):
        st.markdown(user_display_message)
        if attachment and attachment["kind"] == "image":
            st.image(attachment["data"], width=360)

    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        stream_started = [False]
        if attachment and attachment["kind"] == "image":
            if not local_vision.model_is_available():
                st.error("No encuentro el modelo de vision local moondream en Ollama.")
                answer = None
            else:
                with st.status("Erika esta analizando la imagen en este PC...", expanded=False) as generation_status:
                    try:
                        answer = local_vision.analyze_image(
                            attachment["data"], user_message
                        )
                        generation_status.update(label="Analisis listo", state="complete")
                    except requests.RequestException as exc:
                        generation_status.update(label="No se pudo analizar la imagen", state="error")
                        st.error(f"Ollama no pudo analizar la imagen: {exc}")
                        answer = None
                    except Exception as exc:
                        generation_status.update(label="No se pudo analizar la imagen", state="error")
                        st.error(f"Error al analizar la imagen: {exc}")
                        answer = None
        else:
            with st.status("Erika esta pensando...", expanded=False) as generation_status:
                def render_stream(partial: str) -> None:
                    if not stream_started[0]:
                        generation_status.update(label="Erika esta escribiendo...", state="running")
                        stream_started[0] = True
                    answer_placeholder.markdown(partial + " \\u258d")

                if attachment and attachment["kind"] == "document":
                    answer = ask_ollama(
                        selected_model, model_message, False, False,
                        active_conversation_id, on_chunk=render_stream, local_only=True,
                    )
                else:
                    answer = ask_ollama(
                        selected_model, user_message, allow_web, rag_enabled,
                        active_conversation_id, on_chunk=render_stream,
                    )
                if answer:
                    generation_status.update(label="Respuesta lista", state="complete")
                else:
                    generation_status.update(label="No se pudo generar una respuesta", state="error")
        if answer:
            answer_placeholder.markdown(answer)
            save_message("IA", answer, selected_model, active_conversation_id)
    st.rerun()
