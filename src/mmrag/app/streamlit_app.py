"""Thin Streamlit client over the same application pipeline."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from mmrag.pipeline import MMRAGPipeline
from mmrag.schemas import Evidence, Modality

st.set_page_config(page_title="Meridian · MMRAG", page_icon="◉", layout="wide")


@st.cache_resource
def pipeline() -> MMRAGPipeline:
    return MMRAGPipeline.from_config(Path("configs/default.yaml"))


def show_evidence(item: Evidence, show_scores: bool) -> None:
    chunk = item.chunk
    st.caption(f"{chunk.doc_id} · p.{chunk.page} · {chunk.modality.value}")
    if chunk.modality is Modality.TABLE and chunk.raw_grid:
        st.dataframe(chunk.raw_grid, use_container_width=True)
        with st.expander("Serialised table"):
            st.write(chunk.content)
    elif chunk.modality is Modality.FIGURE and chunk.image_path and chunk.image_path.exists():
        st.image(str(chunk.image_path), caption=chunk.caption)
        st.write(chunk.content)
    else:
        st.write(chunk.content)
    if show_scores:
        st.code(f"fused={item.fused_score:.4f} dense={item.dense_score or 0:.4f} bm25={item.bm25_score or 0:.4f}")


st.title("Meridian")
st.caption("Grounded financial document search · research prototype, not financial advice")
with st.sidebar:
    mode = st.selectbox("Mode", ["Multimodal", "Text-only baseline", "Side-by-side"])
    k = st.slider("Evidence count", 1, 20, 5)
    selected_modalities = st.multiselect(
        "Modalities",
        list(Modality),
        default=list(Modality),
        format_func=lambda item: item.value.title(),
    )
    show_scores = st.checkbox("Show retrieval scores", value=False)

examples = [
    "What was revenue in FY2024?",
    "Which table reports operating margin?",
    "What trend is shown in the revenue figure?",
]
question = st.text_input("Question", placeholder=examples[0])
st.caption("Try: " + " · ".join(examples))

if question:
    service = pipeline()
    modality_filter = set(selected_modalities)
    modes = [False, True] if mode == "Side-by-side" else [mode == "Text-only baseline"]
    columns = st.columns(len(modes))
    for column, baseline in zip(columns, modes, strict=True):
        with column, st.spinner("Retrieving citable evidence…"):
            st.subheader("Text-only baseline" if baseline else "Multimodal")
            try:
                answer = service.ask(question, k=k, baseline=baseline, modalities=modality_filter)
            except (RuntimeError, ValueError) as error:
                st.error(str(error))
                continue
            st.success("Grounded ✓" if answer.grounded else "Insufficient evidence")
            st.markdown(answer.text)
            tabs = st.tabs(["All", "Text", "Tables", "Figures"])
            filters = [None, Modality.TEXT, Modality.TABLE, Modality.FIGURE]
            for tab, filter_value in zip(tabs, filters, strict=True):
                with tab:
                    for evidence in answer.evidence:
                        if filter_value is None or evidence.chunk.modality is filter_value:
                            with st.container(border=True):
                                show_evidence(evidence, show_scores)
