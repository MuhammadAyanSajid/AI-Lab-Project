# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
import urllib.parse
from collections import Counter
from bs4 import BeautifulSoup
from search_crawler import run_smart_website_crawler
from ml_classifier import analyze_text_authenticity
from logic_expert import extract_document_metadata, run_logic_inference_engine

# Set Streamlit Page Configurations
st.set_page_config(
    page_title="Researchify - Academic Verifier Engine",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS to improve layout, fonts, and dark theme consistency
st.markdown("""
<style>
    /* Styling headers & backgrounds */
    .stApp {
        background: radial-gradient(circle at 50% 50%, #111827 0%, #090D16 100%);
        color: #F1F5F9;
    }
    .app-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(180deg, rgba(14, 165, 233, 0.05) 0%, rgba(9, 13, 22, 0) 100%);
        border-radius: 16px;
        margin-bottom: 1.5rem;
    }
    .main-title {
        font-family: 'Georgia', serif;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8 0%, #34D399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        margin-bottom: 0.2rem;
        letter-spacing: -0.05rem;
    }
    .subtitle {
        font-family: 'Inter', sans-serif;
        color: #94A3B8;
        font-size: 1.1rem;
        font-weight: 400;
        max-width: 800px;
        margin: 0 auto 1.5rem auto;
        line-height: 1.5;
    }
    .section-header {
        font-family: 'Georgia', serif;
        color: #38BDF8;
        border-bottom: 1px solid #1E293B;
        padding-bottom: 0.4rem;
        margin-top: 1.8rem;
        margin-bottom: 1rem;
    }
    .keyword-badge {
        display: inline-block;
        background-color: #1E293B;
        border: 1px solid #38BDF8;
        color: #38BDF8;
        padding: 4px 10px;
        border-radius: 20px;
        margin: 4px;
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Main Application Title Header
st.markdown("""
<div class='app-header'>
    <div class='main-title'>Researchify</div>
    <div class='subtitle'>A Smart Research Assistant for Web Crawling, Fact-Checking, and Document Authenticity Verification</div>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration and API Setup
st.sidebar.markdown("### 🛠️ Configuration Console")
target_url = st.sidebar.text_input("Target URL Domain / Entrypoint", placeholder="e.g. journals.plos.org")
max_pages = st.sidebar.slider("Page Search Limit", min_value=1, max_value=10, value=3, step=1)
max_depth = st.sidebar.slider("Maximum Scraper Depth", min_value=1, max_value=5, value=2, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔑 API Integrations")
gemini_key = st.sidebar.text_input(
    "Google Gemini API Key", 
    type="password", 
    help="Required for summarization, scholarly citations, and text humanization."
)

if gemini_key:
    genai.configure(api_key=gemini_key)  # type: ignore

st.sidebar.markdown("""
---
### 🔬 Triple-Reasoning Architecture
1. **Guided Crawl (Search AI):** Link prioritizing search scraper.
2. **Ensemble Models (ML AI):** Consensus verification across MNB, LR, and DT.
3. **FOL Expert (Logic AI):** Checks metadata landmarks (DOI, citations).
""")

# --- HELPER FUNCTIONS ---
def call_gemini_api(prompt: str) -> str:
    """Safely queries Gemini model with standard fallback handling."""
    if not gemini_key:
        return "Configure your Gemini API key in the sidebar to activate this feature."
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")  # type: ignore
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error executing Gemini query: {str(e)}"

def generate_gemini_summary(text: str) -> str:
    prompt = (
        "Analyze the following text and generate an extremely simple, concise, and easy-to-understand executive summary. "
        "Your output must consist of exactly 5 bullet points. Each bullet point must be a single, short sentence "
        "written in clear, plain English. Avoid technical jargon, complex metaphors, or dense phrasing:\n\n" + text[:8000]
    )
    return call_gemini_api(prompt)

def humanize_article_with_gemini(text: str) -> str:
    prompt = (
        "You are an academic copyeditor. Rewrite and humanize the following scientific text. "
        "Increase stylistic variance (burstiness), eliminate repetitive transitions, improve "
        "natural readability, and preserve all core scientific and technical arguments:\n\n" + text[:8000]
    )
    return call_gemini_api(prompt)

def extract_keywords_fallback(text: str) -> list:
    stopwords = {"the", "a", "and", "of", "to", "in", "is", "for", "on", "that", "it", "with", "as", "this", "are", "by", "an", "be"}
    words = [w.lower().strip(",.?!()[]{}") for w in text.split() if w.lower().strip(",.?!()[]{}").isalpha()]
    filtered = [w for w in words if w not in stopwords and len(w) > 4]
    counter = Counter(filtered)
    return [item[0] for item in counter.most_common(6)]

def extract_keywords_gemini(text: str) -> list:
    if not gemini_key:
        return extract_keywords_fallback(text)
    prompt = f"Extract exactly 6 core technical keywords or topics from this article text. Return ONLY a comma-separated list of those terms:\n\n{text[:3000]}"
    result = call_gemini_api(prompt)
    if "Error" in result:
        return extract_keywords_fallback(text)
    return [term.strip() for term in result.split(",") if term.strip()]

def extract_page_links(raw_html: str, current_url: str) -> str:
    """Extracts, resolves, and formats all external hyperlinks found on the page for clean copying."""
    soup = BeautifulSoup(raw_html, "html.parser")
    links = []
    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href")
        if href and isinstance(href, str):
            full_url = urllib.parse.urljoin(current_url, href)
            # Retain valid HTTP/S links and remove duplicates
            if full_url.startswith("http") and full_url not in links:
                links.append(full_url)
    if not links:
        return "No reference links detected on this page."
    return "\n".join(links)

def generate_all_citations(url: str, text: str) -> str:
    """Generates bibliographic citations across APA, MLA, Chicago, and BibTeX in a single call."""
    prompt = (
        f"Based on the following document context, generate bibliographic citations for this source "
        f"in all of these four formats: APA, MLA, Chicago, and BibTeX. "
        f"Format each format clearly with uppercase headers (e.g. === APA STYLE ===).\n"
        f"Url: {url}\n\nContent Snippet:\n{text[:2500]}"
    )
    return call_gemini_api(prompt)

def audit_logical_fallacies(text: str) -> str:
    """Scans the text for logical fallacies, methodological bias, or scientific overreaches."""
    prompt = (
        "Analyze the following research text for logical fallacies, cognitive biases, or scientific overreaches. "
        "For each issue identified:\n"
        "1. Fallacy/Bias Name: (e.g., Correlation vs Causation, Hasty Generalization, Appeal to Authority)\n"
        "2. Detailed Explanation: A simple, plain-English description of why this is a logical fallacy and how it weakens the argument.\n"
        "3. Text Context: Cite or describe where this pattern occurs in the text.\n\n"
        "Provide exactly 3 critical findings, formatted in clear Markdown. Keep your explanations highly detailed yet incredibly easy to understand.\n\n"
        f"Text Snippet:\n{text[:6000]}"
    )
    return call_gemini_api(prompt)

def export_text_to_pdf(text_content: str) -> bytes:
    """Generates PDF cleanly. Falls back to plain text bytes if FPDF is not installed."""
    try:
        from fpdf import FPDF  # type: ignore
        class CleanPDF(FPDF):
            def header(self):
                self.set_font('Helvetica', 'B', 14)
                self.cell(0, 10, 'Researchify - Humanized Document Export', ln=True, align='C')
                self.ln(5)
            def footer(self):
                self.set_y(-15)
                self.set_font('Helvetica', 'I', 8)
                self.cell(0, 10, f'Page {self.page_no()}', align='C')
                
        pdf = CleanPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=10)
        
        # Clean text to avoid encoding errors in standard Latin-1 PDF output
        clean_text = text_content.encode('latin-1', 'ignore').decode('latin-1')
        for line in clean_text.split('\n'):
            pdf.multi_cell(0, 8, line)  # Positional text parameter ensures legacy/modern FPDF compatibility
            
        return bytes(pdf.output())
    except Exception:
        # Fallback to plain text bytes
        return text_content.encode('utf-8', errors='ignore')

# Persistent Session states dictionary mapping URLs to their rewritten outputs
if "regenerated_articles" not in st.session_state:
    st.session_state.regenerated_articles = {}

run_analysis = st.sidebar.button("Execute Verification Diagnostics", type="primary", use_container_width=True)

# Clear old rewrites on fresh runs
if run_analysis:
    st.session_state.regenerated_articles = {}

has_articles_in_state = bool(st.session_state.get("regenerated_articles", {}))

if run_analysis or has_articles_in_state:
    if not target_url:
        st.warning("⚠️ Enter a valid domain/entrypoint in the sidebar console to initiate evaluations.")
    else:
        # Execute Scraper/Forensics only on new runs
        if "pages_data" not in st.session_state or run_analysis:
            with st.spinner("Processing Web-spidering and heuristics mapping..."):
                pages_data, crawl_metrics = run_smart_website_crawler(target_url, max_pages=max_pages, max_depth=max_depth)
                st.session_state.pages_data = pages_data
                st.session_state.crawl_metrics = crawl_metrics
        else:
            pages_data = st.session_state.pages_data
            crawl_metrics = st.session_state.crawl_metrics
            
        if not pages_data:
            st.error("❌ Crawler could not extract indexable content. Verify target availability.")
        else:
            page_urls = [page["url"] for page in pages_data]
            
            # --- TOP LEVEL WORKSPACE (REGENERATE AT ANY MOMENT) ---
            st.markdown("<h3 class='section-header'>⚡ Researchify Generative Workspace</h3>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown("#### Dynamic Webpage Humanization & PDF Compilation")
                st.write("Rewrites the crawled document layout using the AI model and allows instant PDF exports.")
                
                # Handle selection variables
                selected_url_main_raw = st.selectbox("Select target page node for workspace actions:", page_urls, key="workspace_selector")
                selected_url_main = str(selected_url_main_raw) if selected_url_main_raw else ""
                
                selected_page_main = next(p for p in pages_data if p["url"] == selected_url_main)
                
                col_actions_a, col_actions_b = st.columns([1, 2])
                with col_actions_a:
                    trigger_regen = st.button("🔄 Regenerate Article (Humanized)", use_container_width=True)
                
                if trigger_regen:
                    if not gemini_key:
                        st.error("Please enter a Google Gemini API Key in the sidebar to activate writing capabilities.")
                    else:
                        with st.spinner("Rewriting entire webpage structure to maximize human stylometric variance..."):
                            regen_result = humanize_article_with_gemini(selected_page_main["text"])
                            # Save specifically to the current URL key
                            st.session_state.regenerated_articles[selected_url_main] = regen_result
                            st.success("Article successfully rewritten!")
                            st.rerun()  # Forces immediate redraw of the text area and PDF download button
                
                # Safely render workspace only if this specific URL has been generated
                current_regen_content = st.session_state.regenerated_articles.get(selected_url_main, "")
                
                if current_regen_content:
                    st.text_area("Regenerated Article Workspace", value=current_regen_content, height=200)
                    pdf_bytes = export_text_to_pdf(current_regen_content)
                    st.download_button(
                        label="📥 Download Regenerated Article as PDF",
                        data=pdf_bytes,
                        file_name="researchify_regenerated_document.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            
            # --- WEB SPIDER EVALUATION PANELS ---
            st.markdown("<h3 class='section-header'>🕸️ Heuristic Search AI Trace</h3>", unsafe_allow_html=True)
            col_cr1, col_cr2, col_cr3 = st.columns(3)
            with col_cr1:
                with st.container(border=True):
                    st.metric(label="Crawler Nodes Expanded", value=f"{crawl_metrics['nodes_expanded']} / {max_pages}")
            with col_cr2:
                with st.container(border=True):
                    st.metric(label="Discovered Interlink Graph", value=f"{crawl_metrics['total_urls_found']} links")
            with col_cr3:
                with st.container(border=True):
                    st.metric(label="Web Spider Processing Time", value=f"{crawl_metrics['execution_time_seconds']}s")
                
            # Explicitly handle None return from selectbox to satisfy Pylance
            selected_url_raw = st.selectbox("Select target page node from crawl graph for details:", page_urls, key="details_selector")
            selected_url = str(selected_url_raw) if selected_url_raw else ""
            
            selected_page = next(p for p in pages_data if p["url"] == selected_url)
            
            # Execute downstream forensics
            with st.spinner("Extracting stylometric features & evaluating models..."):
                ml_results = analyze_text_authenticity(selected_page["html"])
                
            metadata = extract_document_metadata(selected_page["text"], selected_page["html"])
            logic_results = run_logic_inference_engine(
                human_probability=ml_results["global_human_probability"],
                has_doi=metadata["has_doi"],
                has_citations=metadata["has_citations"],
                has_scopus_link=metadata["has_scopus_link"],
                ttr=ml_results["type_token_ratio"],
                burstiness=ml_results["sentence_variance"],
                transition_density=ml_results["transition_density"]
            )
            
            # --- GRAPHICAL LOGICAL AUDIT BOARD ---
            st.markdown("<h3 class='section-header'>📊 Automated Authenticity & Logical Diagnosis</h3>", unsafe_allow_html=True)
            col_diag_left, col_diag_right = st.columns([1, 1])
            
            score_color = "#10B981" if logic_results["final_trust_score"] >= 80 else ("#F59E0B" if logic_results["final_trust_score"] >= 50 else "#EF4444")
            
            with col_diag_left:
                with st.container(border=True):
                    st.markdown("<h4 style='margin-top: 0; color: #38BDF8; text-align: center;'>Cumulative Trust Dial</h4>", unsafe_allow_html=True)
                    
                    # Interactive Plotly Gauge Chart for cumulative score
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=logic_results["final_trust_score"],
                        domain={'x': [0, 1], 'y': [0, 1]},
                        number={'font': {'color': score_color, 'size': 50, 'family': 'Georgia'}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#475569'},
                            'bar': {'color': score_color},
                            'bgcolor': '#1E293B',
                            'borderwidth': 1,
                            'bordercolor': '#334155',
                            'steps': [
                                {'range': [0, 45], 'color': 'rgba(239, 68, 68, 0.08)'},
                                {'range': [45, 65], 'color': 'rgba(245, 158, 11, 0.08)'},
                                {'range': [65, 80], 'color': 'rgba(14, 165, 233, 0.08)'},
                                {'range': [80, 100], 'color': 'rgba(16, 185, 129, 0.08)'}
                            ]
                        }
                    ))
                    fig_gauge.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#E2E8F0', family='Inter, sans-serif'),
                        margin=dict(t=10, b=10, l=15, r=15),
                        height=210
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    st.markdown(f"<div style='text-align: center; font-weight: bold; font-size: 1.15rem; color: #E2E8F0;'>Verdict: <span style='color:{score_color};'>{logic_results['status_verdict']}</span></div>", unsafe_allow_html=True)
                
            with col_diag_right:
                with st.container(border=True):
                    st.markdown("<h4 style='margin-top:0; color: #38BDF8;'>Registry & Citations Audit</h4>", unsafe_allow_html=True)
                    
                    col_meta_a, col_meta_b, col_meta_c = st.columns(3)
                    with col_meta_a:
                        st.write("📌 **DOI Identifier**")
                        if metadata["has_doi"]:
                            st.success("Detected")
                        else:
                            st.error("Missing")
                            
                    with col_meta_b:
                        st.write("📚 **Academic Citations**")
                        if metadata["has_citations"]:
                            st.success("Detected")
                        else:
                            st.error("Missing")
                            
                    with col_meta_c:
                        st.write("🏛️ **Authority Registry**")
                        if metadata["has_scopus_link"]:
                            st.success("Detected")
                        else:
                            st.warning("Unverified")
                            
                    st.markdown("<h4 style='margin-top: 1rem; color: #38BDF8;'>Dynamic Model Agreement</h4>", unsafe_allow_html=True)
                    consensus_data = ml_results["model_predictions_consensus"]
                    for model_name, prob_val in consensus_data.items():
                        st.slider(
                            label=f"{model_name.replace('_', ' ')} Probability",
                            min_value=0.0, max_value=1.0, value=float(prob_val), disabled=True, key=f"sl_{model_name}"
                        )

            # --- DYNAMIC API INTEGRATED METADATA: EXECUTIVE SUMMARY & KEYWORDS BOX ---
            st.markdown("<h3 class='section-header'>📓 Document Intel & Topic Modeling</h3>", unsafe_allow_html=True)
            col_intel_left, col_intel_right = st.columns([1, 1])
            
            with col_intel_left:
                with st.container(border=True):
                    st.markdown("<h4 style='margin-top: 0; color: #38BDF8;'>🔑 Target Article Core Keywords</h4>", unsafe_allow_html=True)
                    st.write("Extracted core technical keywords mapping document contents:")
                    
                    with st.spinner("Extracting index topics..."):
                        extracted_topics = extract_keywords_gemini(selected_page["text"])
                        
                    topics_html = []
                    for topic in extracted_topics:
                        topics_html.append(f"<span class='keyword-badge'>🏷️ {topic}</span>")
                    st.markdown(f"<div>{' '.join(topics_html)}</div>", unsafe_allow_html=True)
                    
            with col_intel_right:
                with st.container(border=True):
                    st.markdown("<h4 style='margin-top: 0; color: #38BDF8;'>📋 Executive Summary (Exactly 5 Lines)</h4>", unsafe_allow_html=True)
                    if not gemini_key:
                        st.info("💡 Provide a Gemini API key in the sidebar to generate a 5-line executive summary of this article.")
                        summary_display = "Configure your Gemini API key in the sidebar to activate this feature."
                    else:
                        with st.spinner("Drafting executive abstract..."):
                            summary_display = generate_gemini_summary(selected_page["text"])
                        st.markdown(summary_display)
                    
                    # Copyable Summary Box
                    st.markdown("<h5 style='color: #64748B;'>📥 Copyable Summary</h5>", unsafe_allow_html=True)
                    st.text_area(
                        label="Copyable Executive Summary Output Console", 
                        value=summary_display, 
                        height=120, 
                        label_visibility="collapsed"
                    )

            # --- WARNING AND HUMANIZER BLOCK (UNDER 50% THRESHOLD) ---
            if logic_results["final_trust_score"] < 50:
                st.markdown("<h3 class='section-header'>⚠️ Low Credibility Risk Warning</h3>", unsafe_allow_html=True)
                with st.container(border=True):
                    st.markdown("""
                    <div style='background-color: rgba(239, 68, 68, 0.1); border-left: 5px solid #EF4444; padding: 1rem; border-radius: 8px;'>
                        <strong style='color: #EF4444;'>SECURITY THREAT TRIGGERED:</strong><br/>
                        This document has scored below the threshold of 50%. This signifies a high likelihood of 
                        robotic syntax duplication, automated drafting constructs, or unvetted reference parameters.
                    </div>
                    """, unsafe_allow_html=True)
                    st.write("")
                    
                    col_warn_a, col_warn_b = st.columns([1, 2])
                    with col_warn_a:
                        trigger_warn_humanize = st.button("🤖 Regenerate & Humanize Text", type="primary", use_container_width=True)
                    
                    if trigger_warn_humanize:
                        if not gemini_key:
                            st.error("Gemini API key is required to humanize the document.")
                        else:
                            with st.spinner("Injecting organic stylometric cadence..."):
                                human_result = humanize_article_with_gemini(selected_page["text"])
                                # Save directly to active key and refresh
                                st.session_state.regenerated_articles[selected_url] = human_result
                                st.success("Article successfully humanized and loaded to Workspace above!")
                                st.rerun()

            # --- STYLOMETRIC RADAR CHART & HEATMAP GRID ---
            st.markdown("<h3 class='section-header'>🕵️ Structural Forensics & Linguistic Layout</h3>", unsafe_allow_html=True)
            col_st_left, col_st_right = st.columns([1, 1])
            
            with col_st_left:
                with st.container(border=True):
                    st.markdown("<h4 style='margin-top: 0; color: #38BDF8;'>Multivariate Linguistic Signature</h4>", unsafe_allow_html=True)
                    
                    # Normalize values to a readable 0-1 scale on radar coordinates
                    norm_ttr = float(ml_results["type_token_ratio"])
                    norm_burst = min(float(ml_results["sentence_variance"]), 50.0) / 50.0
                    norm_trans = min(float(ml_results["transition_density"]) * 15.0, 1.0)
                    norm_rep = 1.0 - min(float(ml_results["repetition_ratio"]), 1.0)
                    
                    radar_categories = [
                        'Lexical Diversity (TTR)', 
                        'Stylistic Cadence (Burstiness)', 
                        'AI Transition Density', 
                        'Lexical Variance (Repetition)'
                    ]
                    radar_values = [norm_ttr, norm_burst, norm_trans, norm_rep]
                    
                    # Plotly polar layout spider plot
                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=radar_values,
                        theta=radar_categories,
                        fill='toself',
                        name='Linguistic Profile',
                        line_color='#38BDF8',
                        fillcolor='rgba(56, 189, 248, 0.25)'
                    ))
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, 1], gridcolor='#334155', linecolor='#334155', tickcolor='#475569'),
                            angularaxis=dict(gridcolor='#334155', linecolor='#334155', tickcolor='#475569')
                        ),
                        showlegend=False,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#94A3B8', family='Inter, sans-serif'),
                        margin=dict(t=30, b=30, l=40, r=40),
                        height=280
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
                
            with col_st_right:
                with st.container(border=True):
                    st.markdown("<h4 style='margin-top: 0; color: #38BDF8;'>Syntactic Cadence Console</h4>", unsafe_allow_html=True)
                    st.markdown("""
                    Sentence text blocks colored dynamically by local AI risk evaluation:
                    - <span style="background-color: #991b1b; color: #fecaca; padding: 2px 4px; border-radius: 4px; font-weight:bold;">High Synthetic Risk (> 75%)</span>
                    - <span style="background-color: #78350f; color: #fef3c7; padding: 2px 4px; border-radius: 4px; font-weight:bold;">Mixed Origin (40% - 75%)</span>
                    - <span style="background-color: #065f46; color: #d1fae5; padding: 2px 4px; border-radius: 4px; font-weight:bold;">Human Composition Style (< 40%)</span>
                    """, unsafe_allow_html=True)
                    
                    heatmap_spans = []
                    for item in ml_results["analyzed_sentences"]:
                        sent = item["sentence"]
                        ai_p = item["ai_probability"]
                        
                        if ai_p > 0.75:
                            style_str = "background-color: #991b1b; color: #fecaca; padding: 4px 6px; border-radius: 6px; margin: 3px; display: inline-block; line-height: 1.8;"
                        elif ai_p >= 0.40:
                            style_str = "background-color: #78350f; color: #fef3c7; padding: 4px 6px; border-radius: 6px; margin: 3px; display: inline-block; line-height: 1.8;"
                        else:
                            style_str = "background-color: #065f46; color: #d1fae5; padding: 4px 6px; border-radius: 6px; margin: 3px; display: inline-block; line-height: 1.8;"
                            
                        heatmap_spans.append(f'<span style="{style_str}" title="AI score: {ai_p:.2%}">{sent}</span>')
                        
                    st.markdown(f"""
                    <div style="background-color: #0B0F19; border: 1px solid #1E293B; border-radius: 12px; padding: 1.25rem; height: 210px; overflow-y: auto; font-family: 'Georgia', serif; font-size: 1.15rem;">
                        {" ".join(heatmap_spans)}
                    </div>
                    """, unsafe_allow_html=True)

            # --- EXPANDED RESEARCH ASSISTANT UTILITIES: ALL-IN-ONE CITATIONS, LINK EXTRACTION & DETAILED FALLACY SCANNER ---
            st.markdown("<h3 class='section-header'>🔬 Scholarly Forensics Workbench</h3>", unsafe_allow_html=True)
            col_util_left, col_util_right = st.columns([1, 1])
            
            with col_util_left:
                with st.container(border=True):
                    st.markdown("<h4 style='margin-top: 0; color: #38BDF8;'>📝 All Academic Citations</h4>", unsafe_allow_html=True)
                    st.write("Bibliographic indexing generated simultaneously across APA, MLA, Chicago, and BibTeX styles:")
                    
                    if not gemini_key:
                        citations_output = "Configure your Gemini API key in the sidebar to generate formatted citation codes."
                        st.info("💡 Provide a Gemini API key to generate citation indexes.")
                    else:
                        with st.spinner("Generating citations schema..."):
                            citations_output = generate_all_citations(selected_url, selected_page["text"])
                        st.code(citations_output, language="text")
                    
                    # Copyable Citations Box
                    st.markdown("<h5 style='color: #64748B;'>📥 Copyable Citation Text</h5>", unsafe_allow_html=True)
                    st.text_area(
                        label="Copyable Citations Output Console", 
                        value=citations_output, 
                        height=130, 
                        label_visibility="collapsed"
                    )
                
                with st.container(border=True):
                    st.markdown("<h4 style='margin-top: 0; color: #38BDF8;'>🔗 Extracted Reference Links Box</h4>", unsafe_allow_html=True)
                    st.write("All active external reference hyperlinks identified within the source web structure:")
                    
                    extracted_links = extract_page_links(selected_page["html"], selected_url)
                    
                    # Copyable Links Box
                    st.text_area(
                        label="Copyable Reference Links Output Console", 
                        value=extracted_links, 
                        height=150, 
                        label_visibility="collapsed"
                    )
                        
            with col_util_right:
                with st.container(border=True):
                    st.markdown("<h4 style='margin-top: 0; color: #38BDF8;'>🔍 Fallacy & Scientific Bias Scanner</h4>", unsafe_allow_html=True)
                    st.write("In-depth analysis scanning the document for cognitive biases, logical arguments, and structural flaws:")
                    
                    if not gemini_key:
                        fallacies_output = "Configure your Gemini API key in the sidebar to run logical evaluations."
                        st.info("💡 Provide a Gemini API key to check logical arguments and identify potential research biases.")
                    else:
                        with st.spinner("Auditing research arguments for logical anomalies..."):
                            fallacies_output = audit_logical_fallacies(selected_page["text"])
                        st.markdown(fallacies_output)
                    
                    # Copyable Fallacy Box
                    st.markdown("<h5 style='color: #64748B;'>📥 Copyable Fallacy Audit</h5>", unsafe_allow_html=True)
                    st.text_area(
                        label="Copyable Fallacies Output Console", 
                        value=fallacies_output, 
                        height=180, 
                        label_visibility="collapsed"
                    )
                
            # --- EVALUATION METRICS AND INTERFACE LOG TABS ---
            st.markdown("<h3 class='section-header'>📊 Technical Metrics & System Diagnostics</h3>", unsafe_allow_html=True)
            tab_ml, tab_logic, tab_raw = st.tabs(["🤖 Multi-Algorithm Comparison Matrix", "⚙️ Prolog-Style Rule Trace Console", "📄 Clean Scraped Text"])
            
            with tab_ml:
                col_met1, col_met2 = st.columns([1, 1])
                with col_met1:
                    st.markdown("<h4 style='color: #38BDF8;'>Statistical Metrics Output</h4>", unsafe_allow_html=True)
                    st.write(f"📊 **Consensus Human Score:** {ml_results['global_human_probability']:.2%}")
                    st.write(f"📊 **Consensus AI Score:** {ml_results['global_ai_probability']:.2%}")
                    st.write(f"📈 **Type-Token Ratio (Lexical Diversity):** {ml_results['type_token_ratio']}")
                    st.write(f"📈 **Burstiness (Sentence Length Variance):** {ml_results['sentence_variance']}")
                    st.write(f"📈 **AI Transition Word Density:** {ml_results['transition_density']:.2%}")
                    st.write(f"📈 **Lexical Repetition Factor:** {ml_results['repetition_ratio']:.2%}")
                    st.write(f"📈 **Mean Sentence Length:** {ml_results['mean_sentence_length']} words")
                with col_met2:
                    st.markdown("<h4 style='color: #38BDF8;'>AI Algorithm Comparison Benchmark</h4>", unsafe_allow_html=True)
                    benchmark_df = pd.DataFrame(ml_results["benchmarks"]).T
                    st.table(benchmark_df)
                    
            with tab_logic:
                st.markdown("<h4 style='color: #38BDF8;'>Rule Engine Deductions</h4>", unsafe_allow_html=True)
                st.code(logic_results["logic_trace"], language="prolog")
                
            with tab_raw:
                st.markdown("<h4 style='color: #38BDF8;'>Whitespace-Compressed Page Content</h4>", unsafe_allow_html=True)
                st.text_area("Extracted Plain Text", value=selected_page["text"], height=300, disabled=True, key="plain_text_preview")

else:
    st.info("👋 Enter a starting domain inside the sidebar panel and click **Execute Verification Diagnostics** to initiate evaluations.")
    st.markdown("""
    ### 🔬 Researchify: Academic Verification Protocol
    
    Researchify integrates three layers of reasoning to scan, verify, and authenticate digital research papers:
    
    1. **Guided Search Engine AI:** Prioritizes crawling resources containing semantic academic keywords (e.g., *abstract*, *journal*, *doi*) while filtering out static media.
    2. **Machine Learning Ensemble:** Runs token distributions through **Multinomial Naive Bayes**, **Logistic Regression**, and **Decision Tree** models to compute a weighted consensus risk.
    3. **Logical Reasoning Expert:** Passes metadata flags and deep stylometrics (Burstiness, Type-Token Ratios) through a declarative rule matrix to establish a unified credibility rating.
    """)