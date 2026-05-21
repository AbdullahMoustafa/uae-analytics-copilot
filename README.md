# UAE Economic Indicators Knowledge Copilot

A conversational, tool-using agent that answers questions about United Arab Emirates
economic indicators — definitions, time series, cross-source reconciliation, and
ad-hoc analysis — directly from World Bank and IMF open data.

Built on Llama 3.3 70B via [Groq](https://groq.com) (free tier), with a
[DuckDB](https://duckdb.org) warehouse, a [ChromaDB](https://www.trychroma.com)
vector store for semantic search, and a curated cross-source lineage graph.

---

## Capabilities

| Capability | What the agent does |
|---|---|
| **Metric search** | Semantic search over ~60 indicator definitions, with units, methodology, and source organization |
| **Time-series retrieval** | Pulls any UAE indicator's full annual history from the warehouse |
| **Cross-source reconciliation** | Side-by-side comparison of World Bank vs IMF values for the same concept, plus curated notes on why they differ |
| **Topic catalog** | Browse the indicator universe by topic (Economy & Growth, Inflation, Trade, Energy, etc.) |
| **Ad-hoc SQL** | The agent writes and executes read-only SQL against the warehouse for computations the other tools don't cover (YoY growth, ranking, correlation, joins) |
| **Lineage lookup** | For any indicator, surface which other sources publish the same concept and the methodology differences |

The agent decides which tool(s) to call per query, executes them, and synthesizes a
cited answer. Every numeric claim is grounded in a `(source, code)` reference.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                       CLI  /  Streamlit UI                           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│              Agent loop  (Llama 3.3 70B via Groq)                    │
│  - 7 function-calling tools  - typed exception handling              │
│  - per-turn trace events     - recovery from malformed tool calls    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
       ┌───────────────────────┼────────────────────────┐
       ▼                       ▼                        ▼
┌──────────────┐       ┌──────────────┐         ┌──────────────┐
│ Vector Store │       │   DuckDB     │         │  Knowledge   │
│  (ChromaDB)  │       │  Warehouse   │         │    Graph     │
│              │       │              │         │  (Lineage)   │
│ Definitions  │       │ Time series  │         │ Source map   │
│ Methodology  │       │ Indicator    │         │ + methodology│
│              │       │ metadata     │         │ divergences  │
└──────┬───────┘       └──────┬───────┘         └──────┬───────┘
       │                      │                        │
       └──────────────────────┴────────────────────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │  Ingestion layer     │
                   │  - World Bank API    │
                   │  - IMF DataMapper    │
                   └──────────────────────┘
```

---

## Setup

### Prerequisites

- Python 3.11+
- A free Groq API key — sign up at [console.groq.com/keys](https://console.groq.com/keys)
- ~500 MB free disk for raw data and embeddings

### Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Configure

Copy [.env.example](.env.example) to `.env` and paste in your Groq key:

```
GROQ_API_KEY=gsk_your-key-here
```

### Build the corpus (one-time, ~5 minutes)

```powershell
python scripts/ingest.py        # pull World Bank + IMF data
python scripts/build_index.py   # warehouse + lineage + vector index
```

### Run

```powershell
python scripts/chat.py                          # CLI
streamlit run src/uae_copilot/app.py            # web UI
```

---

## Example queries

| Query | Tools invoked |
|---|---|
| "What does 'GDP (current US$)' mean and who publishes it?" | `search_definitions` |
| "Show me UAE GDP from 2015 to 2023" | `get_indicator` |
| "Why does UAE GDP differ between World Bank and IMF?" | `compare_sources` |
| "What topics do you cover?" | `list_topics` |
| "What energy indicators do you have?" | `list_indicators_in_topic` |
| "Compute YoY non-oil GDP growth since 2010" | `run_sql` |
| "What's the lineage of the inflation indicator?" | `get_lineage` |
| "Is the UAE diversifying away from oil? Show evidence." | `list_indicators_in_topic` + `run_sql` + `search_definitions` |

---

## Use in data analytics and BI workflows

This system is a **conversational metadata layer** over a structured warehouse. That
makes it a natural fit for several patterns inside an analytics or BI environment.

### 1. Embedded chat panel inside a BI dashboard

Add the Streamlit UI as an iframe in a Tableau, Power BI, or Looker dashboard so
analysts and executives can ask questions about the metrics they're looking at —
*without* leaving the dashboard or pinging the data team.

- **Tableau** — embed via [Tableau Extensions API](https://tableau.github.io/extensions-api/)
- **Power BI** — use a [Custom Visual](https://learn.microsoft.com/en-us/power-bi/developer/visuals/) iframe wrapper
- **Looker Studio** — [Community Visualization](https://developers.google.com/looker-studio/visualization)
- **Apache Superset** — [Custom Plugin Chart](https://superset.apache.org/docs/contributing/howtos/) with embedded chat

Typical questions a user can answer without filing a ticket:
- "What exactly does this KPI tile measure?"
- "Why does this number differ from the figure on Finance's dashboard?"
- "Show me the 10-year trend for the indicator behind this card."

### 2. REST service for a custom analytics frontend

Wrap [`Agent`](src/uae_copilot/agent/agent.py) in a FastAPI service to expose the
agent as an HTTP endpoint. Any frontend (React, Vue, internal portal) can then
POST a question and render the streamed response.

```python
# Minimal pattern — drop into a fastapi app
from fastapi import FastAPI
from uae_copilot.agent.agent import Agent

app = FastAPI()
agent = Agent()  # one shared instance reuses the vector store + warehouse

@app.post("/ask")
def ask(question: str):
    result = agent.run(question)
    return {"answer": result.final_text, "tools_used": [
        e.payload["name"] for e in result.events if e.kind == "tool_call"
    ]}
```

### 3. Drop-in metric governance layer

Replace a wiki-based data dictionary with a queryable one. The agent already enforces:

- **Definitions on every metric** — the methodology note from the source is returned with every search hit
- **Source attribution** — every numeric answer cites `(source, code)`
- **Cross-source divergence flags** — the `compare_sources` tool surfaces conflicts the team would otherwise litigate over Slack

Pair the agent with your existing metric registry (dbt `schema.yml`, Looker LookML,
Cube semantic layer) by adapting [`ingest/`](src/uae_copilot/ingest/) to read from
those files alongside the public sources.

### 4. Direct BI tool connection to the warehouse

The agent doesn't have to be the only consumer of the data it pulls. The DuckDB
warehouse at [data/warehouse.duckdb](data) is queryable from any BI tool that
supports DuckDB (Tableau via ODBC, Power BI via custom connector, Apache Superset
natively, or Hex/Mode notebooks via the duckdb Python driver).

This lets you:
- Build traditional dashboards on top of the same `indicators` and `observations` tables
- Use the agent as a sidecar for context that doesn't fit on a chart (definitions, lineage, "why is this off")
- Avoid duplicating ingestion logic

### 5. Slack / Teams bot for self-service Q&A

Bind the agent to a Slack slash-command. Analysts post `/uae compare UAE current
account WB vs IMF` and the bot responds with the cited answer plus a link to the
underlying observations. Reduces interruptions to the central analytics team.

### 6. Jupyter / Hex / Mode notebook helper

Import [`Agent`](src/uae_copilot/agent/agent.py) directly into a notebook to use
it as an in-line research assistant during exploratory analysis. The agent's
`run_sql` results come back as structured data you can pipe into pandas, matplotlib,
or downstream cells.

---

## Project layout

```
PRO RAG/
├── src/uae_copilot/
│   ├── config.py             - Pydantic settings, paths, model selection
│   ├── ingest/               - Source-specific data pullers (WB, IMF)
│   ├── knowledge/            - Metric dictionary, lineage graph, topic catalog
│   ├── storage/              - DuckDB warehouse wrapper
│   ├── retrieval/            - Sentence-transformers embeddings + ChromaDB
│   ├── agent/                - System prompt, tool schemas, agent loop
│   ├── cli.py                - Typer CLI
│   └── app.py                - Streamlit UI
├── scripts/                  - ingest / build_index / chat entry points
├── data/                     - raw/, processed/, chroma/, warehouse.duckdb (gitignored)
└── tests/                    - smoke tests
```

---

## Provider notes

**Groq** was chosen as the LLM provider for its free tier (~14,400 requests/day
on `llama-3.3-70b-versatile`, ~12K tokens-per-minute) and OpenAI-compatible
function calling. The agent loop in [`agent/agent.py`](src/uae_copilot/agent/agent.py)
is the only file that depends on the provider SDK; switching to OpenAI, Together,
Fireworks, or a local Ollama endpoint requires changing only the client and a few
exception classes.

Embeddings run locally via
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
(~80 MB model, no API key required). Swap for a hosted embedding provider in
[`retrieval/embeddings.py`](src/uae_copilot/retrieval/embeddings.py) if you need
higher recall.

---

## License

MIT
