# genAICourse

Hands-on course code for building generative-AI systems from first principles — tokenizers,
language models, RAG pipelines, transformers, multi-agent setups, and neural networks
implemented from scratch with NumPy.

Each top-level directory is one module of the course. Most files are standalone scripts meant
to be read and run individually rather than imported as a package.

## Repository layout

### `w1/` — Tokenization and a first language model

| File | What it does |
| --- | --- |
| `tokenizer.py` | Minimal demo of Hugging Face `GPT2Tokenizer` encode/decode. |
| `byte_pair_encoding.py` | `BPETokenizer` — byte-pair encoding implemented from scratch (vocabulary init, iterative merges). |
| `basic_language_model.py` | `BasicLanguageModel` — an n-gram style model trained on `data/weather_data.txt`, with a Gradio UI and a 90/10 train/test split. |
| `pytorch_example.py` | A minimal PyTorch loop: custom `Dataset`, `DataLoader`, and a two-layer ReLU MLP. |
| `data/weather_data.txt` | Training corpus for `basic_language_model.py`. |

### `w2/` — Retrieval-augmented generation

**`rag_part_1/` — embeddings and retrieval basics**

| File | What it does |
| --- | --- |
| `rag01.py` | GloVe word vectors via `gensim` — cosine similarity and `king - man + queen` style analogies. |
| `rag02.py` | Book recommender using `sentence-transformers` embeddings and scikit-learn cosine similarity. |
| `rag02_streamlit.py` | The same recommender wrapped in a Streamlit UI. |
| `rag03.py` | `VisualizeChunking` — LangChain `RecursiveCharacterTextSplitter` plus a Chroma vector store, to inspect chunk size/overlap effects. |
| `doc.txt` | Sample document used for the chunking walkthrough. |

**`rag_part_2/` — a full RAG app, embeddings analysis, and reranking**

| File | What it does |
| --- | --- |
| `rag04.py` | `PDFChatApp` — end-to-end conversational RAG over PDFs: `PyPDFLoader` → chunking → HF embeddings → Chroma → `ChatGroq` (`llama-3.3-70b-versatile`) with `ConversationBufferMemory`, served through Gradio. |
| `rag05.py` | `VizEmbedding` — token-level embeddings from `jinaai/jina-embeddings-v2-base-en` projected to 2-D with t-SNE. |
| `rag06.py` | Cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L6-v2`) to reorder retrieved candidates. |

### `w3/` — Transformers and multi-agent systems

| File | What it does |
| --- | --- |
| `transformer_scratch.py` | A GPT built from scratch in PyTorch: causal multi-head attention, transformer blocks, `tiktoken` GPT-2 encoding, plus training and generation loops. |
| `host_agent_chat_ui.py` | Streamlit chat front-end for a multi-agent setup (Demand / Greeting / Orchestrator), posting to an orchestrator at `localhost:10002`. |
| `fixed_host_agent_chat_ui.py` | A revised version of the same chat UI. |
| `streamlit_host_ui_with_a2aclient.py` | The chat UI driven through an `A2AClient` (agent-to-agent protocol). See the note below. |

### `w4/` — Placeholder for the next module (currently empty).

### `nn/` — Neural networks from scratch

| File | What it does |
| --- | --- |
| `NeuralNetworks_Coding_From_Scratch_Part1.ipynb` | Neurons, layers, and forward passes with NumPy. |
| `Coding layers Neural Network.ipynb` | Stacking dense layers into a network. |
| `Backprop_Single_Neuron.ipynb` | Backpropagation derived and coded for one neuron (ReLU, manual gradients, SGD step). |
| `Combined Backprop.ipynb` | Backpropagation across a full multi-layer network. |
| `ADAM Optimizer.ipynb` | The Adam optimizer implemented and compared against plain SGD. |
| `dense_layer.py` | Helper functions shared by the notebooks: `sigmoid`, weight init, and dot-product/manual layer math. |

## Setup

The project targets **Python 3.13** (see `.venv/pyvenv.cfg`).

```bash
python3 -m venv .venv
source .venv/bin/activate
```

There is no dependency manifest checked in yet. Install what a given script imports, for example:

```bash
# w1
pip install torch transformers gradio numpy

# w2
pip install gensim sentence-transformers scikit-learn langchain langchain-chroma \
            langchain-groq langchain-community chromadb pypdf python-dotenv \
            gradio streamlit matplotlib transformers

# w3
pip install torch tiktoken streamlit requests pillow

# nn
pip install numpy jupyter matplotlib
```

### Secrets

`w2/rag_part_2/rag04.py` loads configuration with `python-dotenv` and calls Groq. Create a local
`.env` (it is git-ignored) alongside it:

```
GROQ_API_KEY=your-key-here
```

## Running things

```bash
# plain scripts
python w1/byte_pair_encoding.py
python w3/transformer_scratch.py

# basic_language_model.py reads data/ by a relative path, so run it from w1/
cd w1 && python basic_language_model.py

# Streamlit apps
streamlit run w2/rag_part_1/rag02_streamlit.py
streamlit run w3/host_agent_chat_ui.py

# notebooks
jupyter notebook nn/
```

## Notes and known rough edges

- `w1/basic_language_model.py` opens `data/weather_data.txt` via a **relative** path, so it must be
  run from inside `w1/`.
- `w3/streamlit_host_ui_with_a2aclient.py` contains a **hardcoded absolute `sys.path` entry** pointing
  at a separate local `smart-bems` project, and imports `client.client` / `models.agent` from it.
  It will not run as-is on another machine — repoint that path first.
- The `w3` agent UIs expect an orchestrator already listening on `localhost:10002`; that service is
  not part of this repo.
- Several `w2` scripts use older `langchain.*` import paths (`langchain.embeddings`,
  `langchain.vectorstores`, `langchain.document_loaders`). On current LangChain these moved to
  `langchain_community.*` / `langchain_huggingface` and will emit deprecation warnings or fail.
- The `nn/` notebooks embed screenshot attachments, which is why two of them are ~6.5 MB.
