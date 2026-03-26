"""
COM748 — Hybrid Semantic Search for Academic Papers
Streamlit Community Cloud deployment
Files hosted on HuggingFace Datasets (no GitHub file size limit)
"""

import streamlit as st
import pandas as pd
import numpy as np
import faiss
import requests
import re
import time
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Academic Paper Search",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    :root { --accent: #2E7D32; --accent-light: #E8F5E9; }
    .main-title {
        font-size: 2.2rem; font-weight: 700;
        color: #1a1a1a; margin-bottom: 0;
    }
    .subtitle {
        font-size: 1rem; color: #666;
        margin-top: 0.2rem; margin-bottom: 1.5rem;
    }
    .result-card {
        background: #fff;
        border: 1px solid #e0e0e0;
        border-left: 4px solid #2E7D32;
        border-radius: 8px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .paper-title {
        font-size: 1.05rem; font-weight: 600;
        color: #1a1a1a; margin-bottom: 0.3rem;
    }
    .paper-meta {
        font-size: 0.82rem; color: #888;
        margin-bottom: 0.6rem;
    }
    .score-badge {
        display: inline-block;
        background: #E8F5E9; color: #2E7D32;
        border: 1px solid #A5D6A7;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.78rem; font-weight: 600;
        margin-right: 6px;
    }
    .score-badge-grey {
        background: #f5f5f5; color: #666;
        border-color: #ddd;
    }
    .abstract-text {
        font-size: 0.88rem; color: #333;
        line-height: 1.6; margin-top: 0.7rem;
    }
    .attention-label {
        font-size: 0.75rem; color: #999;
        margin-top: 0.5rem; margin-bottom: 0.2rem;
    }
    .rank-number {
        font-size: 1.4rem; font-weight: 700;
        color: #2E7D32; margin-right: 0.5rem;
    }
    .stButton > button {
        background-color: #2E7D32 !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover {
        background-color: #1B5E20 !important;
    }
    .metric-box {
        background: #f9f9f9;
        border: 1px solid #eee;
        border-radius: 6px;
        padding: 0.6rem 1rem;
        text-align: center;
    }
    .metric-val { font-size: 1.3rem; font-weight: 700; color: #2E7D32; }
    .metric-lbl { font-size: 0.75rem; color: #888; }
</style>
""", unsafe_allow_html=True)

# ── Config — UPDATE THESE WITH YOUR HF REPO ──────────────────────────────
HF_DATASET_REPO = "bhuttashinigami/academic-search-data"   # ← change this
HF_TOKEN        = os.environ.get("HF_TOKEN", "")

HF_API_SCIBERT  = "https://api-inference.huggingface.co/pipeline/feature-extraction/allenai/scibert_scivocab_uncased"
HF_API_SPECTER  = "https://api-inference.huggingface.co/pipeline/feature-extraction/allenai/specter"

WEIGHTS   = {"scibert": 0.40, "specter": 0.40, "recency": 0.10, "citation": 0.10}
TOP_K_DEFAULT = 10

# ── Cache directory (Streamlit Cloud persists /tmp between reruns) ────────
CACHE_DIR = Path("/tmp/paper_search_cache")
CACHE_DIR.mkdir(exist_ok=True)


def download_if_needed(filename: str) -> str:
    """Download file from HuggingFace Dataset if not already cached."""
    cached_path = CACHE_DIR / filename
    if cached_path.exists():
        return str(cached_path)

    with st.spinner(f"Downloading {filename} from HuggingFace..."):
        path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=filename,
            repo_type="dataset",
            token=HF_TOKEN if HF_TOKEN else None,
            local_dir=str(CACHE_DIR),
        )
    return path


@st.cache_resource(show_spinner=False)
def load_data():
    """Download and load all search resources. Cached for session lifetime."""

    # Download files from HuggingFace Dataset
    meta_path    = download_if_needed("papers_metadata.csv")
    scibert_path = download_if_needed("scibert.faiss")
    specter_path = download_if_needed("specter.faiss")

    # Load metadata
    df = pd.read_csv(meta_path, low_memory=False)
    df = df.dropna(subset=["abstract"])
    df = df[df["abstract"].str.strip() != ""].reset_index(drop=True)
    df["id"]       = df["id"].astype(int)
    df["published"] = pd.to_datetime(df["published"], errors="coerce", utc=True)
    if "citation_count" not in df.columns:
        df["citation_count"] = 0
    df["citation_count"] = df["citation_count"].fillna(0).astype(int)

    # Load FAISS indices
    sci_index = faiss.read_index(scibert_path)
    spe_index = faiss.read_index(specter_path)

    # Pre-compute bibliometric scores
    years = df["published"].dt.year.fillna(
        df["published"].dt.year.min()).astype(float)
    min_y, max_y = years.min(), years.max()
    if min_y != max_y:
        recency = ((years - min_y) / (max_y - min_y)).to_numpy(dtype=np.float32)
    else:
        recency = np.ones(len(df), dtype=np.float32)

    cites   = df["citation_count"].to_numpy(dtype=np.float32)
    log_c   = np.log1p(cites)
    citation = (log_c / log_c.max()).astype(np.float32) if log_c.max() > 0 \
               else np.zeros(len(df), dtype=np.float32)

    paper_id_at = df["id"].to_numpy(dtype=int)

    return df, sci_index, spe_index, recency, citation, paper_id_at


def encode_via_api(text: str, api_url: str) -> np.ndarray | None:
    """Encode text via HuggingFace Inference API."""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    try:
        resp = requests.post(
            api_url,
            headers=headers,
            json={"inputs": text, "options": {"wait_for_model": True}},
            timeout=40,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        if isinstance(data[0], list):
            vec = np.array(data[0], dtype=np.float32)
            if vec.ndim == 2:
                vec = vec.mean(axis=0)
        else:
            vec = np.array(data, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.reshape(1, -1).astype(np.float32)
    except Exception:
        return None


def search(query, df, sci_index, spe_index,
           recency, citation, paper_id_at, top_k=TOP_K_DEFAULT):
    t0 = time.time()

    sci_vec = encode_via_api(query, HF_API_SCIBERT)
    spe_vec = encode_via_api(query, HF_API_SPECTER)

    if sci_vec is None and spe_vec is None:
        return [], 0.0

    pool = min(top_k * 15, 300)
    sci_map, spe_map = {}, {}

    if sci_vec is not None:
        scores, positions = sci_index.search(sci_vec, pool)
        sci_map = {int(p): float(s) for p, s in zip(positions[0], scores[0])}
    if spe_vec is not None:
        scores, positions = spe_index.search(spe_vec, pool)
        spe_map = {int(p): float(s) for p, s in zip(positions[0], scores[0])}

    results = []
    for pos in set(sci_map) | set(spe_map):
        if pos >= len(df):
            continue
        sci_sim = sci_map.get(pos, 0.0)
        spe_sim = spe_map.get(pos, 0.0)
        rec_sc  = float(recency[pos])
        cite_sc = float(citation[pos])
        score   = (0.40*sci_sim + 0.40*spe_sim +
                   0.10*rec_sc  + 0.10*cite_sc)
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
            "citation_score": round(cite_sc, 4),
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results[:top_k], (time.time() - t0) * 1000


def attention_highlight_html(abstract: str, query: str) -> str:
    """Lightweight attention proxy — highlights query-relevant words."""
    query_tokens = set(re.sub(r"[^a-z0-9 ]", " ", query.lower()).split())
    words  = abstract.split()
    scores = []
    for word in words:
        clean = re.sub(r"[^a-z0-9]", "", word.lower())
        if clean in query_tokens:
            scores.append(1.0)
        elif any(qt in clean or clean in qt
                 for qt in query_tokens if len(qt) > 3):
            scores.append(0.6)
        elif len(clean) > 6:
            scores.append(0.2)
        else:
            scores.append(0.05)

    max_s  = max(scores) if max(scores) > 0 else 1.0
    scores = [s / max_s for s in scores]

    parts = []
    for word, score in zip(words, scores):
        if score > 0.5:
            op = 0.15 + score * 0.45
            parts.append(
                f'<span style="background:rgba(46,125,50,{op:.2f});'
                f'border-radius:3px;padding:1px 2px">{word}</span>')
        elif score > 0.15:
            op = score * 0.5
            parts.append(
                f'<span style="background:rgba(255,193,7,{op:.2f});'
                f'border-radius:3px;padding:1px 2px">{word}</span>')
        else:
            parts.append(word)
    return " ".join(parts)


def format_date(dt) -> str:
    try:
        return dt.strftime("%b %Y")
    except Exception:
        return "—"


def render_card(rank, paper, query, show_attention):
    authors = paper["authors"]
    authors_str = ""
    if authors and authors != "nan" and authors.strip():
        authors_str = f"&nbsp;·&nbsp; 👤 {authors[:80]}{'...' if len(authors)>80 else ''}"

    st.markdown(f"""
    <div class="result-card">
        <div style="display:flex;align-items:flex-start;gap:0.8rem;">
            <span class="rank-number">#{rank}</span>
            <div style="flex:1;">
                <div class="paper-title">{paper['title']}</div>
                <div class="paper-meta">
                    📅 {format_date(paper['published'])}{authors_str}
                </div>
                <span class="score-badge">Score {paper['final_score']:.3f}</span>
                <span class="score-badge score-badge-grey">SciBERT {paper['scibert_sim']:.3f}</span>
                <span class="score-badge score-badge-grey">SPECTER {paper['specter_sim']:.3f}</span>
                <span class="score-badge score-badge-grey">Recency {paper['recency_score']:.3f}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if show_attention:
        preview = paper["abstract"][:400] + ("..." if len(paper["abstract"]) > 400 else "")
        attn_html = attention_highlight_html(preview, query)
        st.markdown(f"""
            <div class="attention-label">
                🔍 Abstract — highlighted words indicate query relevance
            </div>
            <div class="abstract-text">{attn_html}</div>
        """, unsafe_allow_html=True)
    else:
        preview = paper["abstract"][:400] + "..."
        st.markdown(f'<div class="abstract-text">{preview}</div>',
                    unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Read full abstract"):
        st.write(paper["abstract"])


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    st.markdown('<div class="main-title">🔬 Academic Paper Search</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Hybrid semantic search · 25,000 CS papers · '
        'SciBERT + SPECTER + Recency · COM748 Research Project</div>',
        unsafe_allow_html=True)

    # Load (downloads from HF on first run, cached after)
    with st.spinner("Loading search index (first load may take ~30 seconds)..."):
        try:
            df, sci_index, spe_index, recency, citation, paper_id_at = load_data()
        except Exception as e:
            st.error(f"Failed to load index: {e}\n\nMake sure HF_DATASET_REPO is set correctly in app.py")
            st.stop()

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        top_k         = st.slider("Results to return", 5, 20, TOP_K_DEFAULT)
        show_attention = st.toggle("Show attention highlighting", value=True)

        st.markdown("---")
        st.markdown("### 📊 Corpus")
        c1, c2 = st.columns(2)
        c1.markdown(f'<div class="metric-box"><div class="metric-val">{len(df):,}</div>'
                    f'<div class="metric-lbl">Papers</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-box"><div class="metric-val">2</div>'
                    f'<div class="metric-lbl">Models</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🏗️ Architecture")
        st.markdown("""
        **Retrieval:** FAISS IndexFlatIP  
        **Models:** SciBERT + SPECTER  
        **Scoring:**  
        · 40% SciBERT cosine  
        · 40% SPECTER cosine  
        · 10% Publication recency  
        · 10% Citation count*  
        
        *Unavailable via arXiv API
        """)

        st.markdown("---")
        st.markdown("### 💡 Example Queries")
        examples = [
            "transformer attention mechanisms NLP",
            "federated learning differential privacy",
            "graph neural networks node classification",
            "knowledge distillation model compression",
            "contrastive learning sentence embeddings",
            "generative adversarial image synthesis",
            "reinforcement learning policy optimization",
            "named entity recognition information extraction",
        ]
        for eq in examples:
            if st.button(eq, use_container_width=True, key=f"ex_{eq[:15]}"):
                st.session_state["query_input"] = eq
                st.rerun()

    # Search bar
    query = st.text_input(
        "Search",
        placeholder="e.g. transformer models for scientific text classification...",
        key="query_input",
        label_visibility="collapsed",
    )
    search_clicked = st.button("🔍 Search", use_container_width=False)

    # Run search
    if search_clicked and query.strip():
        with st.spinner("Encoding query and searching... (first query may take ~15 seconds)"):
            results, latency_ms = search(
                query.strip(), df, sci_index, spe_index,
                recency, citation, paper_id_at, top_k=top_k
            )

        if not results:
            st.warning("No results. The HuggingFace Inference API may be warming up — please try again in 20 seconds.")
            st.stop()

        # Metrics row
        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        for col, val, lbl in [
            (m1, str(len(results)),          "Results"),
            (m2, f"{latency_ms:.0f}ms",      "Latency"),
            (m3, f"{results[0]['final_score']:.3f}", "Top Score"),
            (m4, f"{len(df):,}",             "Corpus"),
        ]:
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
                "SciBERT":     [r["scibert_sim"] for r in results],
                "SPECTER":     [r["specter_sim"] for r in results],
            }, index=[f"#{i}" for i in range(1, len(results)+1)])
            st.line_chart(chart_df)

    elif search_clicked and not query.strip():
        st.warning("Please enter a search query.")
    else:
        st.markdown("---")
        st.markdown("""
        <div style="text-align:center;padding:3rem;color:#aaa;">
            <div style="font-size:3rem;">🔬</div>
            <div style="font-size:1.1rem;margin-top:0.5rem;">
                Enter a query above or pick an example from the sidebar
            </div>
            <div style="font-size:0.85rem;margin-top:0.5rem;">
                SciBERT + SPECTER transformer models · 25,000 CS papers
            </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()