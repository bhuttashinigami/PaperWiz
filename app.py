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
    :root {
        --primary: #2E7D32;
        --primary-soft: #E8F5E9;
        --bg: #f4f6f9;
        --panel: white;
        --text: #212121;
        --muted: #6f7a8a;
        --border: #d9e1e8;
    }
    .stApp {
        background: linear-gradient(180deg, #f4f6f9 0%, #eef2f7 100%);
        color: var(--text);
    }
    .main-title {
        font-size: 2.4rem; font-weight: 800;
        color: var(--text); margin-bottom: .1rem;
        letter-spacing: .4px;
    }
    .subtitle {
        font-size: 1.05rem; color: var(--muted);
        margin-top: .2rem; margin-bottom: 1.5rem;
    }
    .result-card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-left: 4px solid var(--primary);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 24px rgba(33, 43, 54, 0.08);
        transition: transform .18s ease, box-shadow .18s ease;
    }
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 28px rgba(33, 43, 54, 0.14);
    }
    .paper-title {
        font-size: 1.1rem; font-weight: 700;
        color: #111; margin-bottom: .3rem;
    }
    .paper-meta { font-size: .83rem; color: #727f8f; margin-bottom: .55rem; }
    .score-badge {
        display: inline-flex;
        align-items: center;
        gap: .2rem;
        background: var(--primary-soft); color: var(--primary);
        border: 1px solid #c8e6c9;
        border-radius: 999px;
        padding: 3px 10px; font-size: .78rem;
        font-weight: 700; margin-right: 6px;
    }
    .score-badge-grey {
        background: #f6f7fb; color: #4d596b; border-color: #dfe7f1;
    }
    .abstract-text {
        font-size: .9rem; color: #2f3d4c;
        line-height: 1.65; margin-top: .75rem;
    }
    .attention-label {
        font-size: .77rem; color: #546a82;
        margin-top: .5rem; margin-bottom: .2rem;
        font-style: italic;
    }
    .rank-number {
        font-size: 1.45rem; font-weight: 800;
        color: var(--primary);
        min-width: 2.1rem;
    }
    .stButton > button {
        background-color: var(--primary) !important;
        color: #fff !important; border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        padding: 8px 16px !important;
    }
    .stButton > button:hover {
        filter: brightness(1.04);
    }
    .stSidebar .css-1d391kg {
        background: linear-gradient(180deg, #ffffff 0%, #f9fbfd 100%);
        border-radius: 12px;
        padding: 12px;
    }
    .metric-box {
        background: #ffffff; border: 1px solid #e2e8f0;
        border-radius: 10px; padding: .65rem .9rem; text-align: center;
    }
    .metric-val { font-size: 1.35rem; font-weight: 800; color: var(--primary); }
    .metric-lbl { font-size: .74rem; color: #6e7a8b; letter-spacing: .3px; }
    .streamlit-expanderHeader {
        color: #2f4256 !important;
        font-weight: 700 !important;
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
    st.info(f"Downloading {filename}...")
    path = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename=filename,
        repo_type="dataset",
        token=HF_TOKEN if HF_TOKEN else None,
        local_dir=str(CACHE_DIR),
    )
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
            parts.append(f'<span style="background:rgba(46,125,50,{op:.2f});'
                         f'border-radius:3px;padding:1px 2px">{word}</span>')
        elif s > 0.15:
            op = s * 0.5
            parts.append(f'<span style="background:rgba(255,193,7,{op:.2f});'
                         f'border-radius:3px;padding:1px 2px">{word}</span>')
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
          <div class="paper-meta">📅 {fmt_date(paper['published'])}{auth_str}</div>
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
            <div class="attention-label">Abstract — green = query-relevant terms</div>
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
    st.markdown(
        '<div class="subtitle">Hybrid semantic search · 25,000 CS papers · '
        'SciBERT + SPECTER · COM748 Masters Project</div>',
        unsafe_allow_html=True)

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