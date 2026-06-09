# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from search_crawler import run_smart_website_crawler
from ml_classifier import analyze_text_authenticity
from logic_expert import extract_document_metadata, run_logic_inference_engine

# Set Streamlit Page Configurations
st.set_page_config(
    page_title="Aletheia - Academic Verifier Engine",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced Custom CSS for a cyber-scholarly dark laboratory environment
st.markdown("""
<style>
    /* Dark void styling across page body */
    .stApp {
        background: radial-gradient(circle at 50% 50%, #111827 0%, #090D16 100%);
        color: #F1F5F9;
    }
    
    /* Elegant Title and Badges */
    .app-header {
        text-align: center;
        padding: 2.5rem 0 1rem 0;
        background: linear-gradient(180deg, rgba(14, 165, 233, 0.05) 0%, rgba(9, 13, 22, 0) 100%);
        border-radius: 16px;
        margin-bottom: 2rem;
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
    
    /* Academic Lab Cards with Soft Glow Borders */
    .premium-card {
        background-color: #131A2A;
        border: 1px solid #1E293B;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 4px 20px 0 rgba(14, 165, 233, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .premium-card:hover {
        border-color: #38BDF8;
        box-shadow: 0 6px 24px 0 rgba(14, 165, 233, 0.12);
    }
    
    /* Stepper Progress bar pipeline styling */
    .step-pipeline {
        display: flex;
        justify-content: center;
        gap: 1.5rem;
        margin: 1.5rem 0;
        font-size: 0.9rem;
        color: #64748B;
    }
    .step-node {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .step-active {
        color: #38BDF8;
        font-weight: bold;
    }
    
    /* Header layouts */
    .section-header {
        font-family: 'Georgia', serif;
        color: #38BDF8;
        border-bottom: 1px solid #1E293B;
        padding-bottom: 0.4rem;
        margin-top: 1.8rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Clean CSS Custom Scrollbars for heatmap console */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #090D16;
    }
    ::-webkit-scrollbar-thumb {
        background: #1E293B;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #38BDF8;
    }
</style>
""", unsafe_allow_html=True)

# Application Header & Process Workflow
st.markdown("""
<div class='app-header'>
    <div class='main-title'>Aletheia (ἀλήθεια)</div>
    <div class='subtitle'>Tri-Component Academic Security Engine for Scholarly Verification & Synthesized Content Forensic Analysis</div>
    <div class='step-pipeline'>
        <div class='step-node step-active'>🌐 1. Guided Web Crawl</div>
        <div>➔</div>
        <div class='step-node step-active'>🤖 2. Ensemble Stylometric Machine Learning</div>
        <div>➔</div>
        <div class='step-node step-active'>⚙️ 3. Logical Expert Audit</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Control Center
st.sidebar.markdown("### 🛠️ Lab Configuration Console")
target_url = st.sidebar.text_input("Target URL Domain / Entrypoint", placeholder="e.g. journals.plos.org")
max_pages = st.sidebar.slider("Page Search Limit (Nodes expanded)", min_value=1, max_value=10, value=3, step=1)
max_depth = st.sidebar.slider("Maximum Scraper Path Depth", min_value=1, max_value=5, value=2, step=1)

st.sidebar.markdown("""
---
### 🔬 Triple-Reasoning Strategy
1.  **Guided Search (Component 1):** Best-First Search ranking engine focused on content-heavy structural indicators.
2.  **Ensemble Models (Component 2):** Simultaneous evaluation across Naive Bayes, Linear Models, and Decision Trees.
3.  **First-Order Logic (Component 3):** Verification of metadata landmarks (DOI, citations) and syntactic properties.
""")

run_analysis = st.sidebar.button("Execute Verification Diagnostics", type="primary", use_container_width=True)

if run_analysis:
    if not target_url:
        st.warning("⚠️ Enter a valid domain/entrypoint in the sidebar console to initiate evaluations.")
    else:
        with st.spinner("Processing Web-spidering and heuristics mapping..."):
            pages_data, crawl_metrics = run_smart_website_crawler(target_url, max_pages=max_pages, max_depth=max_depth)
            
        if not pages_data:
            st.error("❌ Crawler could not extract indexable textual content. Verify target availability.")
        else:
            page_urls = [page["url"] for page in pages_data]
            
            # --- WEB SPIDER EVALUATION PANELS ---
            st.markdown("<h3 class='section-header'>🕸️ Heuristic Search AI Trace</h3>", unsafe_allow_html=True)
            col_cr1, col_cr2, col_cr3 = st.columns(3)
            with col_cr1:
                st.markdown(f"""
                <div class='premium-card' style='text-align: center;'>
                    <h5 style='color: #94A3B8; margin:0;'>Crawler Nodes Expanded</h5>
                    <h2 style='color: #38BDF8; margin: 0.5rem 0;'>{crawl_metrics["nodes_expanded"]} / {max_pages}</h2>
                </div>
                """, unsafe_allow_html=True)
            with col_cr2:
                st.markdown(f"""
                <div class='premium-card' style='text-align: center;'>
                    <h5 style='color: #94A3B8; margin:0;'>Discovered Interlink Graph</h5>
                    <h2 style='color: #14B8A6; margin: 0.5rem 0;'>{crawl_metrics["total_urls_found"]} links</h2>
                </div>
                """, unsafe_allow_html=True)
            with col_cr3:
                st.markdown(f"""
                <div class='premium-card' style='text-align: center;'>
                    <h5 style='color: #94A3B8; margin:0;'>Web Spider Processing Time</h5>
                    <h2 style='color: #10B981; margin: 0.5rem 0;'>{crawl_metrics["execution_time_seconds"]}s</h2>
                </div>
                """, unsafe_allow_html=True)
                
            selected_url = st.selectbox("Select target page node from crawl graph:", page_urls)
            selected_page = next(p for p in pages_data if p["url"] == selected_url)
            
            # Execute downstream forensics
            with st.spinner("Executing ensemble models and structural checks..."):
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
            
            # Determine color parameters for layout
            score_color = "#10B981" if logic_results["final_trust_score"] >= 80 else ("#F59E0B" if logic_results["final_trust_score"] >= 50 else "#EF4444")
            
            with col_diag_left:
                st.markdown("<div class='premium-card' style='height: 380px;'>", unsafe_allow_html=True)
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
                    height=200
                )
                st.plotly_chart(fig_gauge, use_container_width=True)
                st.markdown(f"<div style='text-align: center; font-weight: bold; font-size: 1.15rem; color: #E2E8F0;'>Verdict: <span style='color:{score_color};'>{logic_results['status_verdict']}</span></div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col_diag_right:
                st.markdown("<div class='premium-card' style='height: 380px;'>", unsafe_allow_html=True)
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
                        label=f"{model_name.replace('_', ' ')} Probability Estimation",
                        min_value=0.0, max_value=1.0, value=float(prob_val), disabled=True
                    )
                st.markdown("</div>", unsafe_allow_html=True)
                
            # --- STYLOMETRIC RADAR CHART & HEATMAP GRID ---
            st.markdown("<h3 class='section-header'>🕵️ Structural Forensics & Linguistic Layout</h3>", unsafe_allow_html=True)
            col_st_left, col_st_right = st.columns([1, 1])
            
            with col_st_left:
                st.markdown("<div class='premium-card' style='height: 500px;'>", unsafe_allow_html=True)
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
                    height=380
                )
                st.plotly_chart(fig_radar, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col_st_right:
                st.markdown("<div class='premium-card' style='height: 500px;'>", unsafe_allow_html=True)
                st.markdown("<h4 style='margin-top: 0; color: #38BDF8;'>Syntactic Cadence Console</h4>", unsafe_allow_html=True)
                st.markdown("""
                Below is a sentence-by-sentence evaluation highlighting local AI model risk.
                - <span style="background-color: #991b1b; color: #fecaca; padding: 2px 6px; border-radius: 4px; font-weight:bold;">High Synthetic Risk (> 75%)</span>
                - <span style="background-color: #78350f; color: #fef3c7; padding: 2px 6px; border-radius: 4px; font-weight:bold;">Mixed Origin (40% - 75%)</span>
                - <span style="background-color: #065f46; color: #d1fae5; padding: 2px 6px; border-radius: 4px; font-weight:bold;">Human Composition Style (< 40%)</span>
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
                <div style="background-color: #0B0F19; border: 1px solid #1E293B; border-radius: 12px; padding: 1.25rem; height: 310px; overflow-y: auto; font-family: 'Georgia', serif; font-size: 1.15rem;">
                    {" ".join(heatmap_spans)}
                </div>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            # --- EVALUATION METRICS AND INTERFACE LOG TABS ---
            st.markdown("<h3 class='section-header'>🔬 Performance Evaluation & Diagnostics Trace</h3>", unsafe_allow_html=True)
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
                st.text_area("Extracted Plain Text", value=selected_page["text"], height=300, disabled=True)

else:
    st.info("👋 Enter a starting domain inside the sidebar panel and click **Execute Verification Diagnostics** to initiate evaluations.")
    st.markdown("""
    ### 🔬 Multi-Component Integrity Verification Architecture
    
    Aletheia integrates three distinct, coordinated layers of machine reasoning to verify digital research papers:
    
    1.  **Guided Search Engine AI:** Prioritizes crawling resources containing semantic academic keywords (e.g., *abstract*, *journal*, *doi*) while filtering out non-text static media files.
    2.  **Machine Learning Ensemble:** Runs token distributions through **Multinomial Naive Bayes**, **Logistic Regression**, and **Decision Tree** models to compute a weighted consensus risk.
    3.  **Logical Reasoning Expert:** Passes metadata flags and deep stylometrics (Burstiness, Type-Token Ratios) through a declarative rule matrix to establish a unified credibility rating.
    """)