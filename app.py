# app.py
import streamlit as st
import pandas as pd
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

# Custom Styling inject to create an elegant scholarly atmosphere
st.markdown("""
<style>
    .main-title {
        font-family: 'Times New Roman', serif;
        font-weight: 700;
        color: #1E3A8A;
        font-size: 2.8rem;
        margin-bottom: 0.1rem;
    }
    .subtitle {
        font-family: 'Helvetica', sans-serif;
        color: #4B5563;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .section-header {
        font-family: 'Times New Roman', serif;
        color: #1E3A8A;
        border-bottom: 2px solid #E5E7EB;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Application Header Elements
st.markdown("<div class='main-title'>Aletheia (ἀλήθεια)</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>A Tri-Component AI Research Assistant for Intelligent Web Crawling and Academic Document Authenticity Verification</div>", unsafe_allow_html=True)

# Sidebar Control Center
st.sidebar.header("🛠️ Crawl Configuration")
target_url = st.sidebar.text_input("Target URL Domain / Entrypoint", placeholder="https://arxiv.org/ or journals.sagepub.com")
max_pages = st.sidebar.slider("Maximum Pages to Crawl", min_value=1, max_value=10, value=3, step=1)
max_depth = st.sidebar.slider("Maximum Directory Depth", min_value=1, max_value=5, value=2, step=1)

st.sidebar.markdown("""
---
### 📖 How Aletheia Functions
1. **Component 1 (Search AI):** Executes a Heuristic Best-First Search of the target domain. It ranks candidate links using content indicators.
2. **Component 2 (ML Classifier):** Sanitizes html body, splits text to sentence tokens, measures stylistic **Burstiness** (variance), and runs Multinomial Naive Bayes text inference.
3. **Component 3 (Logic Engine):** Performs First-Order Logic validation of academic tags (DOI, citations, Scopus) to construct a verifiable credibility score.
""")

# Action Trigger Button
run_analysis = st.sidebar.button("Run Diagnostic Audit", type="primary", use_container_width=True)

if run_analysis:
    if not target_url:
        st.warning("⚠️ Please provide a valid domain or entrypoint URL in the Sidebar configuration to start.")
    else:
        with st.spinner("Executing Intelligent Search & Structural Analysis..."):
            # Step 1: Run Search Crawler
            pages_data, crawl_metrics = run_smart_website_crawler(target_url, max_pages=max_pages, max_depth=max_depth)
            
        if not pages_data:
            st.error("❌ Crawler was unable to retrieve any text pages from the specified URL. Please verify network access, robots.txt restrictions, or URL parameters and try again.")
        else:
            st.success(f"Successfully scraped and analyzed {len(pages_data)} pages within target scope.")
            
            # For demonstration, we aggregate or analyze the primary/deepest page found, or let the user choose which crawled page to inspect.
            page_urls = [page["url"] for page in pages_data]
            
            st.markdown("<h2 class='section-header'>🌐 Intelligent Web Crawling Trace</h2>", unsafe_allow_html=True)
            col_c1, col_c2, col_c3 = st.columns(3)
            with col_c1:
                st.metric(label="Nodes Expanded (Pages Crawled)", value=crawl_metrics["nodes_expanded"])
            with col_c2:
                st.metric(label="Total Relative URLs Discovered", value=crawl_metrics["total_urls_found"])
            with col_c3:
                st.metric(label="Crawler Crawl-Time (Seconds)", value=f"{crawl_metrics['execution_time_seconds']}s")
                
            # Allow user to toggle between crawled pages
            selected_url = st.selectbox("Select crawled page for detailed linguistic and logical analysis:", page_urls)
            selected_page = next(p for p in pages_data if p["url"] == selected_url)
            
            # Step 2: Machine Learning Evaluation on selected page content
            with st.spinner("Analyzing stylistic patterns & calculating probabilities..."):
                ml_results = analyze_text_authenticity(selected_page["html"])
                
            # Step 3: Extract scholarly metadata
            metadata = extract_document_metadata(selected_page["text"], selected_page["html"])
            
            # Step 4: Run FOL Rule Inference
            logic_results = run_logic_inference_engine(
                human_probability=ml_results["global_human_probability"],
                has_doi=metadata["has_doi"],
                has_citations=metadata["has_citations"],
                has_scopus_link=metadata["has_scopus_link"]
            )
            
            # --- MAIN CREDIBILITY MATRIX PANEL ---
            st.markdown("<h2 class='section-header'>🎯 Core Authenticity & Logic Matrix</h2>", unsafe_allow_html=True)
            col_res_1, col_res_2 = st.columns([1, 2])
            
            with col_res_1:
                # Gauge representation using metric styling
                score_color = "#22C55E" if logic_results["final_trust_score"] >= 80 else ("#EAB308" if logic_results["final_trust_score"] >= 50 else "#EF4444")
                st.markdown(f"""
                <div class='metric-card' style='text-align: center; border-left: 5px solid {score_color};'>
                    <h3 style='margin: 0; color: #4B5563;'>Cumulative Trust Rating</h3>
                    <h1 style='margin: 0.5rem 0; color: {score_color}; font-size: 3.5rem;'>{logic_results['final_trust_score']}<span style='font-size: 1.2rem; color: #6B7280;'>/100</span></h1>
                    <div style='font-weight: bold; font-size: 1.1rem; color: #1F2937;'>{logic_results['status_verdict']}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col_res_2:
                st.markdown("<h4 style='margin-top:0;'>Scholarly Metadata Flags Detected</h4>", unsafe_allow_html=True)
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
                    st.write("🏛️ **Authority Repository**")
                    if metadata["has_scopus_link"]:
                        st.success("Detected")
                    else:
                        st.warning("Unverified")
                        
                st.markdown("""
                *The Cumulative Trust Rating dynamically combines AI-written probability parameters with metadata proofs. 
                Academic texts are expected to retain persistent identifiers (DOI) and structured references.*
                """)
                
            # --- HEATMAP DISPLAY ---
            st.markdown("<h2 class='section-header'>🔍 Visual Linguistic Authenticity Heatmap</h2>", unsafe_allow_html=True)
            st.markdown("""
            Below is a sentence-by-sentence stylometric probability breakdown of the selected page text. 
            Sentences are colored dynamically based on their local machine learning risk evaluations:
            - <span style="background-color: #ffcccc; color: #5f1111; padding: 2px 4px; border-radius: 4px; font-weight:bold;">High Synthetic Risk (> 75% AI)</span>
            - <span style="background-color: #fff2cc; color: #5f4c11; padding: 2px 4px; border-radius: 4px; font-weight:bold;">Mixed Origin (40% - 75% AI)</span>
            - <span style="background-color: #e2f0d9; color: #1e3f20; padding: 2px 4px; border-radius: 4px; font-weight:bold;">Verifiable Human Style (< 40% AI)</span>
            """, unsafe_allow_html=True)
            
            heatmap_html = []
            for item in ml_results["analyzed_sentences"]:
                sent = item["sentence"]
                ai_prob = item["ai_probability"]
                
                if ai_prob > 0.75:
                    # Soft Red
                    style_str = "background-color: #ffcccc; color: #5f1111; padding: 3px 5px; border-radius: 4px; margin: 3px; display: inline-block; line-height: 1.5;"
                    tooltip = f"AI Risk: {ai_prob:.1%}"
                elif ai_prob >= 0.40:
                    # Soft Yellow
                    style_str = "background-color: #fff2cc; color: #5f4c11; padding: 3px 5px; border-radius: 4px; margin: 3px; display: inline-block; line-height: 1.5;"
                    tooltip = f"AI Risk: {ai_prob:.1%}"
                else:
                    # Soft Green
                    style_str = "background-color: #e2f0d9; color: #1e3f20; padding: 3px 5px; border-radius: 4px; margin: 3px; display: inline-block; line-height: 1.5;"
                    tooltip = f"AI Risk: {ai_prob:.1%}"
                    
                heatmap_html.append(f'<span style="{style_str}" title="{tooltip}">{sent}</span>')
                
            st.markdown(f"""
            <div style="background-color: white; border: 1px solid #E5E7EB; border-radius: 8px; padding: 1.5rem; max-height: 400px; overflow-y: auto; font-family: 'Times New Roman', serif; font-size: 1.15rem;">
                {" ".join(heatmap_html)}
            </div>
            """, unsafe_allow_html=True)
            
            # --- AI METRICS PANEL ---
            st.markdown("<h2 class='section-header'>📊 Technical Model Analytics & Evaluation Metrics</h2>", unsafe_allow_html=True)
            tab_ml, tab_logic, tab_raw = st.tabs(["🤖 Machine Learning Performance Profile", "⚙️ Prolog-Style Rule Trace Logs", "📄 Clean Text Content"])
            
            with tab_ml:
                col_m1, col_m2 = st.columns([1, 1])
                with col_m1:
                    st.markdown("#### Global Statistical Stylometry")
                    st.write(f"📊 **Global AI-Generated Probability:** {ml_results['global_ai_probability']:.2%}")
                    st.write(f"🧑 **Global Human-Written Probability:** {ml_results['global_human_probability']:.2%}")
                    st.write(f"📈 **Structural Linguistic Variance (Burstiness):** {ml_results['sentence_variance']}")
                    st.markdown("""
                    *Linguistic Burstiness measures the variation in sentence lengths. Human writers naturally construct 
                    sentences of highly variable lengths (higher variance, e.g., > 15.0), whereas AI outputs display flat, 
                    highly uniform lengths (low variance).*
                    """)
                with col_m2:
                    st.markdown("#### Model Benchmarks (Naive Bayes vs. KNN Baseline)")
                    benchmark_df = pd.DataFrame(ml_results["benchmarks"]).T
                    st.table(benchmark_df)
                    
            with tab_logic:
                st.markdown("#### First-Order Logic Rule Execution Trace")
                st.text_area("Trace Console Log", value=logic_results["logic_trace"], height=250, disabled=True)
                
            with tab_raw:
                st.markdown("#### Sanitized Plaintext Extracted Page Text")
                st.text_area("Extracted Plain Text", value=selected_page["text"], height=250, disabled=True)

else:
    # Warm Welcome landing page state before clicking execute
    st.info("👋 Enter a starting domain/entrypoint in the sidebar panel and click **Run Diagnostic Audit** to perform intelligent crawling, linguistic evaluations, and rule-based authenticity trace audits.")
    st.markdown("""
    ### 🔬 Scientific Integrity Verification
    Aletheia combines multiple distinct technical paradigms to verify if online text is a legitimate, peer-vetted human document or a synthesized machine generation.
    
    * **Heuristic Web Spider (Search AI):** Focuses the spidering process entirely on scientific content by applying a best-first scoring algorithm on relative URL matches.
    * **Stylometric Variance Mapping (ML AI):** Computes sentence structures, statistical variance ratios, and uses CountVectorizers paired with Naive Bayes probability metrics.
    * **First-Order Logic Trace (Prolog Rule Emulation):** Maps factual presence (DOIs, citation counts, scholarly publications) to verify document authority under rigorous rules.
    """)