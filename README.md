Markdown
# 🤖 Erika: Chatbot d'Intel·ligència Artificial Local (3B RAG + SQLite)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Ollama](https://img.shields.io/badge/Ollama-Llama_3.2_(3B)-black.svg)](https://ollama.com/)
[![SQLite](https://img.shields.io/badge/Database-SQLite-003B57.svg)](https://www.sqlite.org/)

> **Treball de Recerca de Batxillerat · Tecnologia**  
> **Autor:** Joel Randy De La Cruz Reynaga  
> **Model principal:** Llama 3.2 (3B) executat de forma 100% local mitjançant Ollama.

---

## 📌 Descripció del Projecte

**Erika** és un assistent conversacional basat en Intel·ligència Artificial dissenyat per executar-se localment en un ordinador personal sense dependre de serveis al núvol ni subscripcions de pagament. El projecte avalua la viabilitat, latència, precisió i capacitat de record d'un model de llenguatge de 3.000 milions de paràmetres (3B) integrat amb un sistema de **Generació Augmentada per Recuperació (RAG)** i **memòria persistent**.

### ✨ Característiques Principals

- **🔒 Privacitat i Sobrania Digital:** Processament de text, tokenització i inferència 100% local (les dades no surten de l'equip).
- **🧠 Memòria Persistent (SQLite):** Detecció automàtica i emmagatzematge de preferències, dades personals i historial de converses.
- **🌐 Cerca Web en Temps Real (RAG):** Mòdul de cerca automatitzat amb Tavily / SerpAPI (i fallback a DuckDuckGo / Bing) per respondre preguntes d'actualitat i citar fonts en temps real.
- **💻 Interfície Web en Streamlit:** Disseny en mode fosc, personalitzat i intuïtiu amb gestió de xats anteriors.
- **🧮 Raonament i Resolució de Problemes:** Optimitzat per a la resolució pas a pas d'exercicis de ciències, matemàtiques i física (ESO, PAU i nivell universitari).

---

## 🏗️ Arquitectura del Sistema

```text
┌──────────────────────────────────────────────────────────┐
│              Interfície d'Usuari (Streamlit)              │
└────────────────────────────┬─────────────────────────────┘
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
┌───────────────────────┐         ┌───────────────────────┐
│ Cerca Web / RAG (API) │         │   Memòria Persistent  │
│ (Tavily/SerpAPI/DDG)  │         │    (Base SQLite)      │
└───────────┬───────────┘         └───────────┬───────────┘
            │                                 │
            └────────────────┬────────────────┘
                             │ (Injecció de Context)
                             ▼
┌──────────────────────────────────────────────────────────┐
│                 Motor d'Inferència Local                 │
│                   Ollama - Llama 3.2 (3B)                │
└──────────────────────────────────────────────────────────┘
🚀 Requisits i Instal·lació
1. Requisits del Sistema
Sistema Operatiu: Windows, macOS o Linux.

Processador: CPU de 4+ nuclis (recomanat GPU dedicada NVIDIA/AMD per a menor latència).

RAM: Mínim 8 GB (Recomanat 16 GB o més).

Python: Versió 3.10 o superior.

Ollama: Instal·lat i configurat a l'equip (Descarregar Ollama).

2. Descarregar el Model
Obre una terminal i executa el següent comandament per descarregar el model Llama 3.2 de 3B:

Bash
ollama pull llama3.2
3. Clonar el Repositori i Instal·lar Dependències
Bash
# Clonar aquest repositori
git clone [https://github.com/Jdc4322/Erika.git](https://github.com/Jdc4322/Erika.git)
cd Erika

# Crear i activar un entorn virtual (opcional però recomanat)
python -m venv venv
# En Windows:
venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate

# Instal·lar les llibreries requerides
pip install -r requirements.txt
⚙️ Configuració i Execució
(Opcional) Claus d'API per a cerca web millorada:

Crea un fitxer .env a la arrel del projecte o configura les variables d'entorn per a Tavily o SerpAPI:

Fragmento de código
TAVILY_API_KEY=la_teva_clau_aqui
SERPAPI_API_KEY=la_teva_clau_aqui
(Si no es configuren claus, el sistema utilitzarà els motors de cerca lliures de fallback).

Iniciar l'aplicació:

Pots executar l'aplicació mitjançant el fitxer per lots en Windows:

DOS
run_app.bat
O directament des de la consola amb Streamlit:

Bash
streamlit run app.py
📊 Paràmetres d'Inferència de l'Experiment
Durant la fase d'avaluació empírica del Treball de Recerca, el sistema es va configurar amb els següents paràmetres de control:

Model: llama3.2:latest (3B)

Temperatura: 0.0 (respostes deterministes)

Finestra de Context: 3072 tokens

Màxim de Tokens Generats: 768 tokens

📜 Llicència i Crèdits
Projecte desenvolupat per Joel Randy De La Cruz Reynaga com a Treball de Recerca de Batxillerat (Curs 2025–2026).
