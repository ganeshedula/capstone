"""UI helper functions, charting, and card formatting for the Streamlit web interface."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st


def pct(value: Any) -> str:
    """Format floating point score (0..1) as percentage string."""
    if value is None or value == "":
        return "—"
    try:
        val = float(value)
        return f"{round(val * 100)}%"
    except (ValueError, TypeError):
        return "—"


def render_header(device_str: str, enn_loaded: bool, rel_loaded: bool):
    """Render top brand header and hardware/checkpoint status pill bar."""
    st.markdown(
        """
        <div class="neural-header">
            <span class="neural-brand-mark">✳</span>
            <h1 class="neural-title">Neural Interactive Console</h1>
        </div>
        <div class="neural-subtitle">
            DoLa Contrastive Decoding · Epistemic Neural Networks · Self-Consistency & RAG Factuality
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    device_badge = f"<span class='status-pill'><span class='status-dot-active'></span><strong>Device:</strong> {device_str.upper()}</span>"
    model_badge = "<span class='status-pill'><span class='status-dot-active'></span><strong>Base Model:</strong> TinyLlama 1.1B</span>"
    enn_badge = (
        "<span class='status-pill'><span class='status-dot-active'></span><strong>ENN Checkpoint:</strong> Loaded</span>"
        if enn_loaded
        else "<span class='status-pill'><span class='status-dot-idle'></span><strong>ENN Checkpoint:</strong> Not Found</span>"
    )
    rel_badge = (
        "<span class='status-pill'><span class='status-dot-active'></span><strong>Reliability ENN:</strong> Trained</span>"
        if rel_loaded
        else "<span class='status-pill'><span class='status-dot-idle'></span><strong>Reliability ENN:</strong> Heuristic</span>"
    )

    st.markdown(
        f"<div class='status-bar'>{model_badge}{device_badge}{enn_badge}{rel_badge}</div>",
        unsafe_allow_html=True,
    )


def render_decision_card(decision: str, reason: str = ""):
    """Render high-visibility ACCEPT or CAUTION banner."""
    is_accept = decision == "ACCEPT"
    css_class = "decision-banner-accept" if is_accept else "decision-banner-caution"
    badge_class = "decision-badge-accept" if is_accept else "decision-badge-caution"
    icon = "✓" if is_accept else "⚠️"
    label = "VERIFIED / ACCEPT" if is_accept else "CAUTION / UNCERTAIN"
    
    clean_reason = reason or ("All configured verification thresholds passed." if is_accept else "Potential uncertainty or low consistency detected.")
    
    st.markdown(
        f"""
        <div class="{css_class}">
            <span class="{badge_class}">{icon} {label}</span>
            <span class="decision-reason">{clean_reason}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_selected_answer(final_res: Optional[Dict[str, Any]]):
    """Render the selected answer prominently without exposing its score."""
    answer = final_res.get("answer", "—") if final_res else "—"

    st.markdown(
        f"""
        <div class="selected-answer-card">
            <div class="selected-answer-heading">
                <span>Selected final answer</span>
            </div>
            <div class="selected-answer-text">{_escape_html(str(answer))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _escape_html(value: str) -> str:
    """Escape answer text before placing it in the highlighted HTML card."""
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def plot_token_diagnostics(token_probabilities: List[float], token_entropies: Optional[List[float]] = None, dola_layers: Optional[List[int]] = None):
    """Create interactive line chart of token probabilities and entropy curve."""
    if not token_probabilities:
        st.info("No token diagnostics available for this response.")
        return
        
    records = []
    for idx, prob in enumerate(token_probabilities):
        rec = {
            "Token Index": idx + 1,
            "Confidence": round(float(prob), 4),
        }
        if token_entropies and idx < len(token_entropies):
            rec["Entropy"] = round(float(token_entropies[idx]), 4)
        if dola_layers and idx < len(dola_layers):
            rec["DoLa Layer"] = dola_layers[idx]
        records.append(rec)
        
    df = pd.DataFrame(records)
    
    try:
        import altair as alt
        chart_data = df.melt("Token Index", value_vars=[c for c in ["Confidence", "Entropy"] if c in df.columns],
                             var_name="Metric", value_name="Score")
        chart = (
            alt.Chart(chart_data)
            .mark_line(point=True)
            .encode(
                x=alt.X("Token Index:Q", title="Generated Token Index"),
                y=alt.Y("Score:Q", title="Metric Value (0 - 1)"),
                color=alt.Color("Metric:N", scale=alt.Scale(domain=["Confidence", "Entropy"], range=["#10b981", "#f59e0b"])),
                tooltip=["Token Index", "Metric", "Score"],
            )
            .properties(height=260)
            .interactive()
        )
        st.altair_chart(
            chart,
            width="stretch",
        )
    except Exception:
        # Fallback to streamlit native line_chart
        st.line_chart(df.set_index("Token Index")[["Confidence"]])


def build_candidates_dataframe(candidates: List[Dict[str, Any]]) -> pd.DataFrame:
    """Transform candidate reports into clean DataFrame."""
    rows = []
    for idx, item in enumerate(candidates):
        metrics = item.get("metrics", {})
        factuality = item.get("factuality", {})
        rows.append({
            "Candidate": f"#{idx + 1}",
            "Rank": item.get("rank", idx + 1),
            "Final Score": pct(item.get("final_score", item.get("score"))),
            "Confidence": pct(item.get("mean_token_confidence", item.get("confidence"))),
            "Consensus": pct(metrics.get("semantic_consistency")),
            "Factuality": pct(metrics.get("factuality_score")),
            "Reliability": pct(metrics.get("reliability_probability")),
            "Hallucination Risk": pct(metrics.get("hallucination_score")),
            "Verifier Status": item.get("verifier_status", factuality.get("verifier_status", "UNKNOWN")),
            "Answer Preview": (item.get("answer", "")[:90] + "...") if len(item.get("answer", "")) > 90 else item.get("answer", ""),
        })
    return pd.DataFrame(rows)


def render_rag_snippets(evidence: List[str]):
    """Render WikiRag passages with source metadata."""
    if not evidence:
        st.info("No WikiRag evidence was retrieved for this prompt.")
        return

    st.markdown(f"**Retrieved {len(evidence)} WikiRag passage(s):**")
    for idx, item in enumerate(evidence):
        if isinstance(item, dict):
            snippet = str(item.get("text", ""))
            source = str(item.get("source", "Source unavailable"))
            url = str(item.get("url", ""))
            kind = str(item.get("kind", "knowledge_base")).replace("_", " ").title()
        else:
            snippet, source, url, kind = str(item), "Source unavailable", "", "Evidence"
        with st.expander(f"📚 {kind} [{idx + 1}]", expanded=(idx == 0)):
            st.caption(f"Source: {source}")
            st.write(snippet)
            if url.startswith(("https://", "http://")):
                st.link_button("Open source", url)
            elif url:
                st.caption(url)


def export_conversation_to_json(history: List[Dict[str, Any]]) -> str:
    """Serialize conversation history to formatted JSON string."""
    exportable = []
    for turn in history:
        clean_turn = {
            "role": turn.get("role"),
            "content": turn.get("content"),
            "mode": turn.get("mode"),
            "decision": turn.get("decision"),
            "metrics": turn.get("metrics"),
        }
        exportable.append(clean_turn)
    return json.dumps(exportable, indent=2)


def export_conversation_to_markdown(history: List[Dict[str, Any]]) -> str:
    """Format conversation history into Markdown report."""
    md_lines = ["# Neural Interactive Session Report\n"]
    for idx, turn in enumerate(history):
        role = turn.get("role", "").capitalize()
        content = turn.get("content", "")
        mode = turn.get("mode", "Default")
        decision = turn.get("decision", "")
        
        md_lines.append(f"### {idx + 1}. {role} ({mode})")
        if decision:
            md_lines.append(f"> **Decision:** {decision}")
        md_lines.append(f"\n{content}\n")
    return "\n".join(md_lines)
