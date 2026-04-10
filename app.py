"""
COM748 — Hybrid Semantic Search for Academic Papers
Loads models locally — no HuggingFace Inference API dependency
"""

import streamlit as st
import pandas as pd
import numpy as np
import faiss
import re
import time
import os
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PaperWiz — Academic Search",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&display=swap');

    :root {
        --primary: #00E5FF;
        --primary-soft: rgba(0, 229, 255, 0.1);
        --bg: #030509;
        --panel: rgba(12, 16, 24, 0.6);
        --text: #e0e7ff;
        --muted: #8292a8;
        --border: rgba(0, 229, 255, 0.15);
        --glow: drop-shadow(0 0 10px rgba(0, 229, 255, 0.4));
    }
    
    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif !important;
    }
    
    .stApp {
        background: radial-gradient(circle at top right, #0a1120 0%, #030509 100%);
        color: var(--text);
    }
    
    /* Global scrollbar styling for futuristic look */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: rgba(0, 229, 255, 0.3); border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--primary); }

    .main-title {
        font-size: 3rem; 
        font-weight: 800;
        color: #fff; 
        margin-bottom: .2rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        text-shadow: 0 0 20px rgba(0, 229, 255, 0.6);
    }
    
    .subtitle {
        font-size: 1.1rem; 
        color: var(--primary);
        margin-top: .2rem; 
        margin-bottom: 2rem;
        font-weight: 300;
        letter-spacing: 0.5px;
        opacity: 0.8;
    }
    
    .result-card {
        background: var(--panel);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--border);
        border-left: 3px solid var(--primary);
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all .25s ease;
    }
    
    .result-card:hover {
        transform: translateY(-3px) scale(1.01);
        border-color: rgba(0, 229, 255, 0.4);
        box-shadow: 0 12px 40px rgba(0, 229, 255, 0.15);
        background: rgba(18, 24, 34, 0.75);
    }
    
    .paper-title {
        font-size: 1.25rem; 
        font-weight: 700;
        color: #ffffff; 
        margin-bottom: .4rem;
        letter-spacing: 0.2px;
    }
    
    .paper-meta { 
        font-size: .85rem; 
        color: var(--muted); 
        margin-bottom: .8rem; 
        font-weight: 300;
    }
    
    .score-badge {
        display: inline-flex;
        align-items: center;
        gap: .3rem;
        background: var(--primary-soft); 
        color: var(--primary);
        border: 1px solid rgba(0, 229, 255, 0.3);
        border-radius: 4px;
        padding: 4px 12px; 
        font-size: .75rem;
        font-weight: 600; 
        margin-right: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        box-shadow: inset 0 0 10px rgba(0, 229, 255, 0.05);
    }
    
    .score-badge-grey {
        background: rgba(255, 255, 255, 0.03); 
        color: #94a3b8; 
        border-color: rgba(255, 255, 255, 0.1);
    }
    
    .abstract-text {
        font-size: .95rem; 
        color: #cbd5e1;
        line-height: 1.7; 
        margin-top: 1rem;
        font-weight: 300;
    }
    
    .attention-label {
        font-size: .75rem; 
        color: #64748b;
        margin-top: .8rem; 
        margin-bottom: .4rem;
        font-style: italic;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .rank-number {
        font-size: 1.6rem; 
        font-weight: 800;
        color: var(--primary);
        min-width: 2.5rem;
        opacity: 0.8;
        text-shadow: 0 0 15px rgba(0, 229, 255, 0.4);
    }
    
    /* Streamlit overrides */
    .stButton > button {
        background: transparent !important;
        color: var(--primary) !important; 
        border: 1px solid var(--primary) !important;
        border-radius: 4px !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.2s ease !important;
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.1) !important;
    }
    
    .stButton > button:hover {
        background: var(--primary-soft) !important;
        box-shadow: 0 0 20px rgba(0, 229, 255, 0.3) !important;
        transform: translateY(-1px);
    }
    
    [data-testid="stSidebar"] {
        background: #020407 !important;
        border-right: 1px solid var(--border);
    }
    
    .metric-box {
        background: rgba(12, 16, 24, 0.4); 
        border: 1px solid var(--border);
        border-radius: 8px; 
        padding: 1rem; 
        text-align: center;
        backdrop-filter: blur(10px);
        transition: border-color .2s ease;
    }
    .metric-box:hover {
        border-color: var(--primary);
    }
    
    .metric-val { 
        font-size: 1.6rem; 
        font-weight: 800; 
        color: var(--primary); 
        text-shadow: 0 0 10px rgba(0, 229, 255, 0.3);
    }
    
    .metric-lbl { 
        font-size: .8rem; 
        color: #94a3b8; 
        letter-spacing: 1px; 
        text-transform: uppercase;
        margin-top: 4px;
    }
    
    .streamlit-expanderHeader {
        color: var(--primary) !important;
        font-weight: 600 !important;
        background: rgba(12, 16, 24, 0.5) !important;
        border-radius: 4px;
        padding-left: 10px !important;
        border: 1px solid transparent;
        transition: border-color .2s ease;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: rgba(0, 229, 255, 0.2);
    }
    
    /* Make inputs look futuristic */
    .stTextInput > div > div > input {
        background: rgba(12, 16, 24, 0.6) !important;
        color: #fff !important;
        border: 1px solid rgba(0, 229, 255, 0.2) !important;
        border-radius: 4px !important;
        padding: 15px !important;
        font-size: 1.1rem !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.2) !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────────────────────
HF_DATASET_REPO = "bhuttashinigami/academic-search-data"  # ← change this
HF_TOKEN        = os.environ.get("HF_TOKEN", "")

SCIBERT_NAME  = "allenai/scibert_scivocab_uncased"
SPECTER_NAME  = "allenai/specter"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

CACHE_DIR = Path("/tmp/paper_search_cache")
CACHE_DIR.mkdir(exist_ok=True)

TOP_K_DEFAULT = 10


# ── Download helpers ──────────────────────────────────────────────────────
def download_if_needed(filename: str) -> str:
    cached = CACHE_DIR / filename
    if cached.exists():
        return str(cached)
    placeholder = st.empty()
    placeholder.info(f"Downloading {filename}...")
    path = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename=filename,
        repo_type="dataset",
        token=HF_TOKEN if HF_TOKEN else None,
        local_dir=str(CACHE_DIR),
    )
    placeholder.empty()
    return path


# ── Load everything once ─────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_all():
    """Download data files + load models. Cached for the session."""

    # ── Data files ────────────────────────────────────────────────────────
    meta_path    = download_if_needed("papers_metadata.csv")
    scibert_path = download_if_needed("scibert.faiss")
    specter_path = download_if_needed("specter.faiss")

    # ── Metadata ──────────────────────────────────────────────────────────
    df = pd.read_csv(meta_path, low_memory=False)
    df = df.dropna(subset=["abstract"])
    df = df[df["abstract"].str.strip() != ""].reset_index(drop=True)
    df["id"]       = df["id"].astype(int)
    df["published"] = pd.to_datetime(df["published"], errors="coerce", utc=True)
    if "citation_count" not in df.columns:
        df["citation_count"] = 0
    df["citation_count"] = df["citation_count"].fillna(0).astype(int)
    paper_id_at = df["id"].to_numpy(dtype=int)

    # ── FAISS indices ─────────────────────────────────────────────────────
    sci_index = faiss.read_index(scibert_path)
    spe_index = faiss.read_index(specter_path)

    # ── Bibliometric scores ───────────────────────────────────────────────
    years = df["published"].dt.year.fillna(2020).astype(float)
    min_y, max_y = years.min(), years.max()
    recency = ((years - min_y) / (max_y - min_y)).to_numpy(dtype=np.float32) \
              if min_y != max_y else np.ones(len(df), dtype=np.float32)

    cites   = df["citation_count"].to_numpy(dtype=np.float32)
    log_c   = np.log1p(cites)
    citation = (log_c / log_c.max()).astype(np.float32) \
               if log_c.max() > 0 else np.zeros(len(df), dtype=np.float32)

    # ── Load transformer models locally ──────────────────────────────────
    # SciBERT
    sci_tok = AutoTokenizer.from_pretrained(SCIBERT_NAME)
    sci_mdl = AutoModel.from_pretrained(SCIBERT_NAME).to(DEVICE)
    sci_mdl.eval()

    # SPECTER
    spe_tok = AutoTokenizer.from_pretrained(SPECTER_NAME)
    spe_mdl = AutoModel.from_pretrained(SPECTER_NAME).to(DEVICE)
    spe_mdl.eval()

    return (df, sci_index, spe_index, recency, citation, paper_id_at,
            sci_tok, sci_mdl, spe_tok, spe_mdl)


# ── Encoding ─────────────────────────────────────────────────────────────
def mean_pool(token_embeddings, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


def encode_query(text: str, tokenizer, model) -> np.ndarray:
    """Encode a query string to a unit-norm vector."""
    with torch.no_grad():
        enc = tokenizer(
            text, padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        )
        input_ids      = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)
        outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
        vec = mean_pool(outputs.last_hidden_state, attention_mask)
        vec = vec.cpu().numpy().astype(np.float32)

    # L2 normalise
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return np.ascontiguousarray(vec)   # shape (1, 768)


# ── Search ────────────────────────────────────────────────────────────────
def search(query, df, sci_index, spe_index, recency, citation, paper_id_at,
           sci_tok, sci_mdl, spe_tok, spe_mdl, top_k=TOP_K_DEFAULT):

    t0 = time.time()
    pool = min(top_k * 15, 300)

    sci_vec = encode_query(query, sci_tok, sci_mdl)
    spe_vec = encode_query(query, spe_tok, spe_mdl)

    sci_scores, sci_pos = sci_index.search(sci_vec, pool)
    spe_scores, spe_pos = spe_index.search(spe_vec, pool)

    sci_map = {int(p): float(s) for p, s in zip(sci_pos[0], sci_scores[0])}
    spe_map = {int(p): float(s) for p, s in zip(spe_pos[0], spe_scores[0])}

    results = []
    for pos in set(sci_map) | set(spe_map):
        if pos >= len(df):
            continue
        sci_sim = sci_map.get(pos, 0.0)
        spe_sim = spe_map.get(pos, 0.0)
        rec_sc  = float(recency[pos])
        cite_sc = float(citation[pos])
        score   = 0.40*sci_sim + 0.40*spe_sim + 0.10*rec_sc + 0.10*cite_sc

        row = df.iloc[pos]
        results.append({
            "paper_id":       int(paper_id_at[pos]),
            "title":          str(row.get("title", "Untitled")),
            "abstract":       str(row.get("abstract", "")),
            "published":      row.get("published"),
            "authors":        str(row.get("authors", "")),
            "final_score":    round(score, 4),
            "scibert_sim":    round(sci_sim, 4),
            "specter_sim":    round(spe_sim, 4),
            "recency_score":  round(rec_sc, 4),
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results[:top_k], (time.time() - t0) * 1000


# ── Attention highlighting ────────────────────────────────────────────────
def attention_html(abstract: str, query: str) -> str:
    query_tokens = set(re.sub(r"[^a-z0-9 ]", " ", query.lower()).split())
    words, scores = abstract.split(), []

    for word in words:
        clean = re.sub(r"[^a-z0-9]", "", word.lower())
        if clean in query_tokens:
            scores.append(1.0)
        elif any(qt in clean or clean in qt for qt in query_tokens if len(qt) > 3):
            scores.append(0.6)
        elif len(clean) > 6:
            scores.append(0.2)
        else:
            scores.append(0.05)

    max_s  = max(scores) if max(scores) > 0 else 1.0
    scores = [s / max_s for s in scores]

    parts = []
    for word, s in zip(words, scores):
        if s > 0.5:
            op = 0.15 + s * 0.45
            parts.append(f'<span style="background:rgba(0,229,255,{op:.2f});'
                         f'color:#fff;border-radius:3px;padding:1px 2px">{word}</span>')
        elif s > 0.15:
            op = s * 0.5
            parts.append(f'<span style="background:rgba(0,180,216,{op:.2f});'
                         f'color:#fff;border-radius:3px;padding:1px 2px">{word}</span>')
        else:
            parts.append(word)
    return " ".join(parts)


def fmt_date(dt) -> str:
    try:    return dt.strftime("%b %Y")
    except: return "—"


def render_card(rank, paper, query, show_attention):
    authors = paper["authors"]
    auth_str = ""
    if authors and authors not in ("nan", ""):
        auth_str = f"&nbsp;·&nbsp; {authors[:80]}{'...' if len(authors)>80 else ''}"

    st.markdown(f"""
    <div class="result-card">
      <div style="display:flex;align-items:flex-start;gap:0.8rem;">
        <span class="rank-number">#{rank}</span>
        <div style="flex:1;">
          <div class="paper-title">{paper['title']}</div>
          <div class="paper-meta">{fmt_date(paper['published'])}{auth_str}</div>
          <span class="score-badge">Score {paper['final_score']:.3f}</span>
          <span class="score-badge score-badge-grey">SciBERT {paper['scibert_sim']:.3f}</span>
          <span class="score-badge score-badge-grey">SPECTER {paper['specter_sim']:.3f}</span>
          <span class="score-badge score-badge-grey">Recency {paper['recency_score']:.3f}</span>
        </div>
      </div>
    """, unsafe_allow_html=True)

    preview = paper["abstract"][:400] + ("..." if len(paper["abstract"]) > 400 else "")
    if show_attention:
        st.markdown(f"""
            <div class="attention-label">Abstract — Cyan = query-relevant terms</div>
            <div class="abstract-text">{attention_html(preview, query)}</div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="abstract-text">{preview}</div>',
                    unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    with st.expander("Read full abstract"):
        st.write(paper["abstract"])


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    st.markdown('<div class="main-title">PaperWiz</div>', unsafe_allow_html=True)
    # st.markdown(
    #     '<div class="subtitle">Hybrid semantic search · 25,000 CS papers · '
    #     'SciBERT + SPECTER · COM748 Masters Project</div>',
    #     unsafe_allow_html=True)

    with st.spinner("Loading models and search index (first load ~60s)..."):
        try:
            (df, sci_index, spe_index, recency, citation, paper_id_at,
             sci_tok, sci_mdl, spe_tok, spe_mdl) = load_all()
        except Exception as e:
            st.error(f"Startup failed: {e}")
            st.info("Make sure HF_DATASET_REPO is set correctly in app.py "
                    "and your data files are uploaded to HuggingFace.")
            st.stop()

    # Sidebar
    with st.sidebar:
        st.markdown("### Settings")
        top_k          = st.slider("Results", 5, 20, TOP_K_DEFAULT)
        show_attention = st.toggle("Attention highlighting", value=True)

        st.markdown("---")
        st.markdown("### Corpus")
        c1, c2 = st.columns(2)
        c1.markdown(f'<div class="metric-box"><div class="metric-val">{len(df):,}</div>'
                    f'<div class="metric-lbl">Papers</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-box"><div class="metric-val">{DEVICE.upper()}</div>'
                    f'<div class="metric-lbl">Device</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Architecture")
        st.markdown("""
        **Retrieval:** FAISS IndexFlatIP  
        **Encoder 1:** SciBERT (scientific domain)  
        **Encoder 2:** SPECTER (citation-informed)  
        **Scoring:**  
        · 40% SciBERT cosine  
        · 40% SPECTER cosine  
        · 10% Publication recency  
        · 10% Citation count*  
        
        *Unavailable via arXiv API
        """)

        st.markdown("---")
        st.markdown("### Try these queries")
        examples = [
            "transformer attention mechanisms NLP",
            "federated learning differential privacy",
            "graph neural networks node classification",
            "knowledge distillation model compression",
            "contrastive learning sentence embeddings",
            "generative adversarial image synthesis",
            "reinforcement learning policy optimization",
            "named entity recognition sequence labelling",
        ]
        for eq in examples:
            if st.button(eq, use_container_width=True, key=f"ex_{eq[:15]}"):
                st.session_state["query_input"] = eq
                st.rerun()

    # Search input
    query = st.text_input(
        "Query",
        placeholder="e.g. transformer models for scientific text classification...",
        key="query_input",
        label_visibility="collapsed",
    )
    search_clicked = st.button("Search")

    if search_clicked and query.strip():
        with st.spinner("Searching..."):
            results, latency_ms = search(
                query.strip(), df, sci_index, spe_index,
                recency, citation, paper_id_at,
                sci_tok, sci_mdl, spe_tok, spe_mdl,
                top_k=top_k,
            )

        if not results:
            st.warning("No results returned. Please try a different query.")
            st.stop()

        # Stats row
        st.markdown("---")
        for col, val, lbl in zip(
            st.columns(4),
            [str(len(results)), f"{latency_ms:.0f}ms",
             f"{results[0]['final_score']:.3f}", f"{len(df):,}"],
            ["Results", "Latency", "Top Score", "Corpus Size"]
        ):
            col.markdown(
                f'<div class="metric-box"><div class="metric-val">{val}</div>'
                f'<div class="metric-lbl">{lbl}</div></div>',
                unsafe_allow_html=True)

        st.markdown(f'\n#### Results for: *"{query}"*')
        st.markdown("---")

        for i, paper in enumerate(results, 1):
            render_card(i, paper, query, show_attention)

        with st.expander("📈 Score distribution"):
            chart_df = pd.DataFrame({
                "Final Score": [r["final_score"] for r in results],
                "SciBERT":     [r["scibert_sim"]  for r in results],
                "SPECTER":     [r["specter_sim"]  for r in results],
            }, index=[f"#{i}" for i in range(1, len(results)+1)])
            st.line_chart(chart_df)

    elif search_clicked:
        st.warning("Please enter a search query.")
    else:
        st.markdown("""
        <div style="text-align:center;padding:3rem;color:#aaa;">
            <div style="font-size:3rem;">🔬</div>
            <div style="font-size:1.1rem;margin-top:0.5rem;">
                Enter a query above or pick an example from the sidebar
            </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()