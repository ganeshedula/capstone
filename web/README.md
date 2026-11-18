# Neural Interactive Web Console (Streamlit)

A modern, inspectable web interface for experimenting with hallucination reduction in LLMs using **DoLa (Decoding by Contrasting Layers)**, **Epistemic Neural Networks (ENN)**, **Self-Consistency with Semantic Clustering**, and **WikiRag Grounding**.

---

## Quickstart

Run with the root launcher:
```bash
./run_web.py
```
Or directly via Streamlit:
```bash
./llm_hallucination_env/bin/python -m streamlit run web/app.py
```
Or through `ask_web.py`:
```bash
./llm_hallucination_env/bin/python ask_web.py --streamlit
```

Open your browser to: **`http://localhost:8501`**

---

## Key Features & Modes

### 1. Execution Modes
- **Comparative (Full Pipeline)** *(Recommended)*:
  Runs Base TinyLlama, DoLa, DoLa + Self-Consistency (with clustering, scoring, and claim verification), and DoLa + ENN side-by-side. Provides an inspectable decision card (`ACCEPT` vs `CAUTION`).
- **DoLa + Self-Consistency**:
  Generates multiple contrastive candidate outputs, clusters them semantically, measures pairwise agreement, and verifies atomic factual claims.
- **DoLa (Contrastive Decoding)**:
  Contrasts logits between mature and premature layers to amplify factual recall and suppress early-layer hallucinations.
- **DoLa + ENN**:
  Incorporates epistemic uncertainty predictions from the trained Epistemic Neural Network into the decoding logits.
- **Base TinyLlama**:
  Greedy or nucleus-sampled autoregressive baseline.

### 2. WikiRag with Qdrant and web context expansion
- **Toggle On/Off**: Enable or disable evidence retrieval from the configured Qdrant collection.
- **Evidence Limit**: Control how many passages are retrieved (1 to 5).
- **Web Context Expansion**: DuckDuckGo results supplement weak or unavailable Qdrant matches by default; set `WIKIRAG_EXPAND_CONTEXT=false` to disable.
- **Configuration**: Set `WIKIRAG_QDRANT_URL`, `WIKIRAG_QDRANT_COLLECTION`, and `WIKIRAG_EMBEDDING_MODEL`; set `WIKIRAG_QDRANT_API_KEY` for authenticated Qdrant. See the root README and `config.yaml` for defaults. The collection must already be vectorized with the same embedding model.
- **RAG Strategies**:
  - *Factuality Verification Only*: Evaluates whether the LLM's generated response is supported or contradicted by retrieved facts.
  - *Context Injection + Verification*: Augments the prompt with WikiRag passages and source attribution so TinyLlama can use the evidence while generating its answer.

### 3. Generation Parameters
- **Max New Tokens**: Slider from 16 to 512 tokens.
- **Temperature & Top-p**: Controls nucleus sampling randomness (set temperature = 0 for greedy decoding).
- **DoLa Alpha ($\alpha$)**: Amplification factor for contrastive layer differences.
- **Self-Consistency Samples**: Number of candidate answers sampled for consensus voting (2 to 10).

### 4. Rich Diagnostics & Visualizations
- **Decision Cards**: High-visibility `VERIFIED / ACCEPT` (Green) or `CAUTION / UNCERTAIN` (Amber) banners.
- **KPI Metrics**: Real-time scores for **Consensus**, **Reliability**, **Hallucination Risk**, and **Factuality**.
- **Interactive Uncertainty Curves**: Altair charts tracking token-by-token confidence and entropy across the generation trajectory.
- **Candidate Breakdown**: Sortable table of all sampled candidate answers with cluster IDs, verifier statuses, and scores.
- **RAG Source Viewer**: Inspect retrieved knowledge-base passages and web results, including available source links.
