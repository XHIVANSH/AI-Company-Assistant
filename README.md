# AI Company Assistant (Python)

One assistant that decides on its own where to find the answer:

| Question type | Tool | Data source |
|---|---|---|
| Latest / real-time info | `web_search` | Tavily |
| "Where is my order?" | `get_order_status` | Order data (mock dict, swap for your API) |
| Company rules (returns, shipping...) | `company_policy_search` | RAG over PDFs in Pinecone |

Stack: Python, Groq LLM, LangChain, Google Gemini Embeddings (free key), Pinecone Vector DB, Tavily Web Search.

## Setup (Python 3.10+)

```
python -m venv .venv
.venv\Scripts\activate          (Windows)   |   source .venv/bin/activate   (Mac/Linux)
pip install -r requirements.txt
copy .env.example .env          (Windows)   |   cp .env.example .env        (Mac/Linux)
```
Fill the 4 keys in `.env`: console.groq.com, aistudio.google.com/apikey, app.pinecone.io, app.tavily.com

## Run

```
python -m src.ingest      # embeds the PDFs in docs/ into Pinecone (run once)
python -m src.assistant   # start chatting
```

Put your PDFs (for example `company_policy.pdf`) in `docs/` and run the ingest command again whenever they change.

## Try these questions

- `What is your return policy?`          -> company_policy_search
- `Where is my order ORD1002?`           -> get_order_status
- `What is the latest news about ISRO?`  -> web_search
- `Hi, who are you?`                     -> no tool
- `Where is my order?`                   -> should ask for the order ID

## Files

- `src/ingest.py` - load PDFs, split, embed, store in Pinecone
- `src/tools.py` - the three tools
- `src/assistant.py` - system prompt, tool-calling loop, chat memory
- `src/config.py` - settings and key checks

If Groq reports an unknown model, set `GROQ_MODEL` in `.env` to a current model name from the Groq console.
