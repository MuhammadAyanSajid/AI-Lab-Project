# search_crawler.py
import time
import requests
import urllib.parse
import heapq
import re
from bs4 import BeautifulSoup

def clean_extra_whitespace_and_lines(text: str) -> str:
    """
    Compresses horizontal spaces/tabs and removes empty vertical carriage returns.
    
    Academic Rationale:
    Raw HTML scrapes contain arbitrary white space from grid alignments, line breaks, 
    and multi-row layout empty sections. Collapsing vertical space to single carriage returns 
    and compressing contiguous spaces to a single space prevents structural layout artifacts 
    from being evaluated as stylistic patterns.
    """
    if not text:
        return ""
    # 1. Convert all horizontal spacing runs (spaces, tabs) into a single space
    text = re.sub(r'[ \t]+', ' ', text)
    # 2. Collapse sequences of multiple empty newlines into a single newline
    text = re.sub(r'\n+', '\n', text)
    # 3. Strip whitespace from individual lines and filter out empty lines
    cleaned_lines = [line.strip() for line in text.split('\n') if line.strip()]
    return '\n'.join(cleaned_lines)

def calculate_url_priority(url: str) -> float:
    """
    Heuristically scores a target URL based on content relevance keywords.
    """
    academic_keywords = [
        "paper", "article", "abstract", "journal", "research", 
        "academic", "thesis", "manuscript", "publication", "science", 
        "document", "pdf", "text", "report", "proceedings"
    ]
    
    score = 0.0
    url_lower = url.lower()
    
    for keyword in academic_keywords:
        if keyword in url_lower:
            score += 10.0
            
    parsed_url = urllib.parse.urlparse(url)
    depth = len([segment for segment in parsed_url.path.split('/') if segment])
    score -= (depth * 1.5)
    
    return score

def run_smart_website_crawler(base_url: str, max_pages: int = 5, max_depth: int = 3) -> tuple:
    """
    Executes a Heuristic/Informed Best-First Search crawler within the base URL's domain.
    
    Returns:
        tuple: (pages_list, metrics_dict)
    """
    start_time = time.time()
    
    if not base_url.startswith("http://") and not base_url.startswith("https://"):
        base_url = "https://" + base_url
        
    parsed_base = urllib.parse.urlparse(base_url)
    base_domain = parsed_base.netloc
    
    priority_queue = []
    initial_priority = calculate_url_priority(base_url)
    heapq.heappush(priority_queue, (-initial_priority, 0, base_url))
    
    visited_urls = set()
    pages_data = []
    total_urls_found = 1
    nodes_expanded = 0
    
    headers = {
        "User-Agent": "AletheiaCrawler/2.0 (+https://github.com/academic/aletheia)"
    }
    
    while priority_queue and len(pages_data) < max_pages:
        neg_priority, depth, current_url = heapq.heappop(priority_queue)
        
        if current_url in visited_urls:
            continue
        visited_urls.add(current_url)
        nodes_expanded += 1
        
        if depth > max_depth:
            continue
            
        try:
            response = requests.get(current_url, headers=headers, timeout=5, allow_redirects=True)
            
            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" not in content_type and "text/plain" not in content_type:
                continue
                
            raw_html = response.text
            soup = BeautifulSoup(raw_html, "html.parser")
            
            # Clean text right at extraction boundary
            extracted_text = soup.get_text()
            sanitized_text = clean_extra_whitespace_and_lines(extracted_text)
            
            pages_data.append({
                "url": current_url,
                "html": raw_html,
                "text": sanitized_text
            })
            
            for anchor in soup.find_all("a", href=True):
                href = anchor.get("href")
                if not href or not isinstance(href, str):
                    continue
                
                full_url = urllib.parse.urljoin(current_url, href)
                parsed_child = urllib.parse.urlparse(full_url)
                
                if parsed_child.netloc == base_domain:
                    path_lower = parsed_child.path.lower()
                    if any(path_lower.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".gif", ".zip", ".mp4", ".mp3", ".css", ".js"]):
                        continue
                        
                    clean_child_url = urllib.parse.urlunparse((
                        parsed_child.scheme, parsed_child.netloc, parsed_child.path, parsed_child.params, parsed_child.query, ""
                    ))
                    
                    if clean_child_url not in visited_urls:
                        total_urls_found += 1
                        p_score = calculate_url_priority(clean_child_url)
                        heapq.heappush(priority_queue, (-p_score, depth + 1, clean_child_url))
                        
        except requests.exceptions.RequestException:
            continue
            
    execution_time = time.time() - start_time
    
    metrics = {
        "nodes_expanded": nodes_expanded,
        "execution_time_seconds": round(execution_time, 4),
        "total_urls_found": total_urls_found
    }
    
    return pages_data, metrics