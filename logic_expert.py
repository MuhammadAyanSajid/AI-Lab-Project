# logic_expert.py
import re

def extract_document_metadata(text: str, raw_html: str) -> dict:
    """
    Scans plain text and raw HTML structure using specific regex patterns to detect academic markers.
    """
    # 1. DOI Pattern
    doi_pattern = re.compile(r'\b10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+\b')
    has_doi = bool(doi_pattern.search(text) or doi_pattern.search(raw_html))
    
    # 2. Citations / Bibliography
    citations_headers = ["references", "bibliography", "works cited", "literature cited"]
    text_lower = text.lower()
    
    has_citations_header = any(header in text_lower for header in citations_headers)
    has_bracket_citations = bool(re.search(r'\[\d{1,3}\]', text))
    has_author_year_citations = bool(re.search(r'\([A-Z][a-zA-Z]+( et al\.)?,\s*\d{4}\)', text))
    
    has_citations = has_citations_header or has_bracket_citations or has_author_year_citations
    
    # 3. Scopus/Publisher Link Detection
    authority_domains = [
        "scopus.com", "doi.org", "researchgate.net", "ncbi.nlm.nih.gov", 
        "orcid.org", "sciencedirect.com", "scholar.google.com", "ieee.org",
        "springer.com", "wiley.com", "arxiv.org"
    ]
    
    has_scopus_link = any(domain in text_lower or domain in raw_html.lower() for domain in authority_domains)
    
    return {
        "has_doi": has_doi,
        "has_citations": has_citations,
        "has_scopus_link": has_scopus_link
    }

def run_logic_inference_engine(
    human_probability: float, 
    has_doi: bool, 
    has_citations: bool, 
    has_scopus_link: bool,
    ttr: float,
    burstiness: float,
    transition_density: float
) -> dict:
    """
    Executes an expanded First-Order Logic (FOL) deduction and trace engine.
    
    Logical Criteria additions:
    Rule A: Evaluates ensemble stylistic human probability score.
    Rule B: Verifies physical global indexing (DOI status).
    Rule C: Verifies academic citations structures.
    Rule D (Vocabulary Richness): Checks Type-Token Ratio. High lexical diversity implies human specialized text.
    Rule E (Monotony Constraint): Penalizes flat sentence-length distributions (low burstiness).
    Rule F (Filler Congruence): High transition density typical of structured AI formats degrades points.
    """
    trust_score = 0
    logic_trace = []
    
    logic_trace.append("--- INITIATING ALETHEIA PROLOG RULE INFERENCE (V2.0) ---")
    logic_trace.append(f"Input States: Human Prob={human_probability:.2f}, DOI={has_doi}, Citations={has_citations}")
    logic_trace.append(f"Stylometrics: TTR={ttr:.4f}, Burstiness={burstiness:.2f}, Transition Density={transition_density:.4f}")
    
    # --- RULE A: Linguistic Stylometry Check (Max 30) ---
    if human_probability >= 0.70:
        trust_score += 30
        logic_trace.append(
            f"RULE A (HUMANITY) FIRED: human_probability ({human_probability:.2%}) >= 70%. "
            "Conclusion: Composite stylistic features correlate with organic human origins. (+30 points)"
        )
    elif human_probability >= 0.40:
        trust_score += 15
        logic_trace.append(
            f"RULE A (HUMANITY) FIRED: human_probability ({human_probability:.2%}) is in the uncertain range. "
            "Conclusion: Potential mixed-source authorship. (+15 points)"
        )
    else:
        logic_trace.append(
            f"RULE A (HUMANITY) FAILED: human_probability ({human_probability:.2%}) is too low. "
            "Conclusion: Lexical patterns suggest synthetic composition. (+0 points)"
        )
        
    # --- RULE B: Persistent DOI Registration (Max 25) ---
    if has_doi:
        trust_score += 25
        logic_trace.append(
            "RULE B (REGISTRY) FIRED: Registered DOI discovered. "
            "Conclusion: Content has a verified tracking record in public scholarly registries. (+25 points)"
        )
    else:
        logic_trace.append(
            "RULE B (REGISTRY) FAILED: No persistent DOI index found. "
            "Conclusion: Content lacks registered metadata identification. (+0 points)"
        )
        
    # --- RULE C: Bibliographic footprint (Max 20) ---
    if has_citations:
        trust_score += 20
        logic_trace.append(
            "RULE C (CITATIONS) FIRED: Bibliographic citations or reference indexes discovered. "
            "Conclusion: Work frames itself within established peer literature structures. (+20 points)"
        )
    else:
        logic_trace.append(
            "RULE C (CITATIONS) FAILED: No active bibliographies detected. "
            "Conclusion: Work lacks documented linkages to standard peer literature. (+0 points)"
        )
        
    # --- RULE D: Lexical Richness / Vocabulary Richness (Max 15) ---
    if ttr >= 0.45:
        trust_score += 15
        logic_trace.append(
            f"RULE D (LEXICAL DIVERSITY) FIRED: Type-Token Ratio ({ttr:.4f}) is exceptionally high. "
            "Conclusion: Extensive semantic complexity, typical of domain-expert writing. (+15 points)"
        )
    elif ttr >= 0.30:
        trust_score += 8
        logic_trace.append(
            f"RULE D (LEXICAL DIVERSITY) FIRED: Type-Token Ratio ({ttr:.4f}) is average. "
            "Conclusion: Normal vocabulary distribution. (+8 points)"
        )
    else:
        logic_trace.append(
            f"RULE D (LEXICAL DIVERSITY) FAILED: Type-Token Ratio ({ttr:.4f}) is low. "
            "Conclusion: High token redundancy detected, characteristic of repetitive AI generation patterns. (+0 points)"
        )
        
    # --- RULE E: Burstiness Rhythm / Monotony constraints (Max 10) ---
    if burstiness >= 15.0:
        trust_score += 10
        logic_trace.append(
            f"RULE E (STYLISTIC CADENCE) FIRED: Sentence-length variance ({burstiness:.2f}) is robust. "
            "Conclusion: Varied sentence pacing suggests human rhythmic expression. (+10 points)"
        )
    else:
        logic_trace.append(
            f"RULE E (STYLISTIC CADENCE) FAILED: Sentence-length variance ({burstiness:.2f}) is flat. "
            "Conclusion: Uniform sentence structure indicates typical AI drafting limits. (+0 points)"
        )
        
    # --- RULE F: AI Transition Filler Word penalty (Constraint Penalty) ---
    if transition_density >= 0.03:
        trust_score -= 10
        logic_trace.append(
            f"RULE F (FILLER PENALTY) FIRED: AI transition density ({transition_density:.2%}) is high. "
            "Conclusion: Overuse of generic connective phrases matches common generative patterns. (-10 points)"
        )
        
    # --- RULE G: Authority Domain Registry bonus ---
    if has_scopus_link:
        trust_score += 5
        logic_trace.append(
            "RULE G (AUTHORITY BONUS) FIRED: Academic indexing domains detected in raw structure. "
            "Conclusion: Page is linked to a reputable registry or scholarly platform. (+5 points)"
        )
        
    # Clamping final rating scale to normal bounds
    final_score = max(0, min(trust_score, 100))
    
    if final_score >= 80:
        status_verdict = "Verified Authentic Research Content"
    elif final_score >= 65:
        status_verdict = "Likely Authentic Human Document (Partial Academic Metadata)"
    elif final_score >= 45:
        status_verdict = "Mixed-Origin Web Content (Caution Recommended)"
    else:
        status_verdict = "High Risk Synthetic Text or Unverified Source"
        
    logic_trace.append(f"FINAL AUDIT RESULT: Score={final_score}/100 | Verdict={status_verdict}")
    logic_trace.append("--- INFERENCE LOOP TERMINATED SUCCESSFULLY ---")
    
    return {
        "final_trust_score": final_score,
        "status_verdict": status_verdict,
        "logic_trace": "\n".join(logic_trace)
    }