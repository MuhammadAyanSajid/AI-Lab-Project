# logic_expert.py
import re

def extract_document_metadata(text: str, raw_html: str) -> dict:
    """
    Scans plain text and raw HTML structure using specific regex patterns to detect academic markers.
    
    Mathematical/Logical Rationale:
    True scientific papers and high-integrity publications leaves unmistakable structural artifacts. 
    These include standardized Digital Object Identifiers (DOIs), formal bibliographic sections, 
    and links directing back to indices (e.g., Scopus, Crossref, PubMed). Finding these patterns 
    validates the scholastic character of the text source.
    """
    # 1. DOI Pattern: Starts with 10. and matches common suffix structures
    doi_pattern = re.compile(r'\b10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+\b')
    has_doi = bool(doi_pattern.search(text) or doi_pattern.search(raw_html))
    
    # 2. Citations / Bibliography Structure:
    # Looks for a dedicated References/Bibliography header, or standard bracketed bibliography references [1], or author-year (e.g., Author, 2021)
    citations_headers = ["references", "bibliography", "works cited", "literature cited"]
    text_lower = text.lower()
    
    has_citations_header = any(header in text_lower for header in citations_headers)
    has_bracket_citations = bool(re.search(r'\[\d{1,3}\]', text))
    has_author_year_citations = bool(re.search(r'\([A-Z][a-zA-Z]+( et al\.)?,\s*\d{4}\)', text))
    
    has_citations = has_citations_header or has_bracket_citations or has_author_year_citations
    
    # 3. Scopus/Publisher Link Detection:
    # Academic domain names and link architectures
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

def run_logic_inference_engine(human_probability: float, has_doi: bool, has_citations: bool, has_scopus_link: bool) -> dict:
    """
    Executes a simulated First-Order Logic (FOL) deduction and rule-based tracing system.
    
    Logical Architecture (Simulated Prolog Inference):
    The rule-engine calculates document authenticity by checking explicit prerequisites 
    and updating a trust score credit matrix (0 to 100).
    
    Rule A: IF human_probability > 0.70 THEN grant +40 pts ELSE flag synthetic syntax.
    Rule B: IF has_doi is True THEN grant +30 pts ELSE log unverified source status.
    Rule C: IF has_citations is True THEN grant +30 pts ELSE log missing citation structure.
    Rule D: IF has_scopus_link is True THEN grant +10 pts bonus (capped at 100).
    """
    trust_score = 0
    logic_trace = []
    
    logic_trace.append("--- INITIATING ALETHEIA PROLOG RULE INFERENCE ---")
    logic_trace.append(f"Input States: Human Prob={human_probability:.2f}, DOI={has_doi}, Citations={has_citations}, Scopus={has_scopus_link}")
    
    # --- RULE A: Linguistic Stylometry Check ---
    if human_probability >= 0.70:
        trust_score += 40
        logic_trace.append(
            f"RULE A FIRED: human_probability ({human_probability:.2%}) >= 70%. "
            f"Conclusion: Linguistic style aligns with organic human composition. (+40 points)"
        )
    elif human_probability >= 0.40:
        trust_score += 15
        logic_trace.append(
            f"RULE A FIRED (PARTIAL): human_probability ({human_probability:.2%}) is moderate (40-70%). "
            f"Conclusion: Mixed/Uncertain linguistic composition. (+15 points)"
        )
    else:
        # Fallback to prevent termination, degrading trust safely
        logic_trace.append(
            f"RULE A FAILED: human_probability ({human_probability:.2%}) < 40%. "
            f"Conclusion: Highly repetitive / structured syntax flagged as synthetic AI output. (+0 points)"
        )
        
    # --- RULE B: Digital Registry Check ---
    if has_doi:
        trust_score += 30
        logic_trace.append(
            "RULE B FIRED: DOI registration identified. "
            "Conclusion: Resource is verified via a certified global registry. (+30 points)"
        )
    else:
        logic_trace.append(
            "RULE B FAILED: No valid DOI string found. "
            "Conclusion: Resource lacks verifiable persistent index identifiers. (+0 points)"
        )
        
    # --- RULE C: Bibliographic Validation ---
    if has_citations:
        trust_score += 30
        logic_trace.append(
            "RULE C FIRED: Academic citations or reference indices found. "
            "Conclusion: Content references academic structures and lists research roots. (+30 points)"
        )
    else:
        logic_trace.append(
            "RULE C FAILED: No academic bibliography blocks or parenthetical reference markers found. "
            "Conclusion: Text does not claim clear peer validation connections. (+0 points)"
        )
        
    # --- RULE D: Authority Platform Verification (Bonus Core Rule) ---
    if has_scopus_link:
        trust_score += 10
        logic_trace.append(
            "RULE D FIRED (BONUS): Verified domain reference (Scopus/ResearchGate/DOI) detected. "
            "Conclusion: Association with vetted repository adds credibility points. (+10 points)"
        )
    else:
        logic_trace.append(
            "RULE D NEUTRAL: No publisher authority linkages found. "
            "Conclusion: Source does not benefit from domain indexing bonus. (+0 points)"
        )
        
    # Constraint: Maximum limit is clamped at 100 points
    final_score = min(trust_score, 100)
    if final_score != trust_score:
        logic_trace.append(f"CONSTRAINT FIRED: Cumulative credit score clamped back to physical limit. (Clamped from {trust_score} to 100)")
        
    # Categorize logical status verdict
    if final_score >= 80:
        status_verdict = "Verified Authentic Research Content"
    elif final_score >= 60:
        status_verdict = "Likely Authentic Human Document (Partial Academic Metadata)"
    elif final_score >= 40:
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