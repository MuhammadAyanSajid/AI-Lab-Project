# search_crawler.py
import time
import requests
import urllib.parse
import heapq
from bs4 import BeautifulSoup

def calculate_url_priority(url: str) -> float:
    """
    Heuristically scores a target URL based on content relevance keywords.
    
    Academic Rationale for Informed Best-First Search:
    Standard search techniques (BFS or DFS) treat all web pages uniformly. However, web crawls 
    contain high-noise zones (e.g., /assets, /css, login, contact-us pages) that consume valuable 
    network bandwidth and time. An Informed Search assigns priority to nodes likely to yield rich text. 
    By checking for academic-centric semantic keywords ('paper', 'article', 'journal', 'abstract') 
    and penalizing deep folder nesting (via slash counts), we convert a brute-force traversal 
    into an optimized, goal-directed heuristic search.
    """
    academic_keywords = [
        "paper", "article", "abstract", "journal", "research", 
        "academic", "thesis", "manuscript", "publication", "science", 
        "document", "pdf", "text", "report", "proceedings"
    ]
    
    score = 0.0
    url_lower = url.lower()
    
    # 1. Award points for matches of academic and content keywords
    for keyword in academic_keywords:
        if keyword in url_lower:
            score += 10.0
            
    # 2. Penalize depth of URL structure. Deep directories usually contain asset fragments or specific media.
    # Higher path complexity degrades score to keep search prioritized on shallow index and content hubs.
    parsed_url = urllib.parse.urlparse(url)
    depth = len([segment for segment in parsed_url.path.split('/') if segment])
    score -= (depth * 1.5)
    
    return score

def run_smart_website_crawler(base_url: str, max_pages: int = 5, max_depth: int = 3) -> tuple:
    """
    Executes a Heuristic/Informed Best-First Search crawler within the base URL's domain.
    
    Returns:
        tuple: (pages_list, metrics_dict)
            pages_list: A list of dicts with keys "url", "html", "text"
            metrics_dict: A dict tracking nodes_expanded, execution_time_seconds, and total_urls_found.
    """
    start_time = time.time()
    
    # Ensure URL is properly formatted
    if not base_url.startswith("http://") and not base_url.startswith("https://"):
        base_url = "https://" + base_url
        
    parsed_base = urllib.parse.urlparse(base_url)
    base_domain = parsed_base.netloc
    
    # Priority Queue elements are tuples of (negative_priority, depth, url)
    # We use negative priority because Python's heapq is a min-heap, and we want high scores popped first.
    priority_queue = []
    initial_priority = calculate_url_priority(base_url)
    heapq.heappush(priority_queue, (-initial_priority, 0, base_url))
    
    visited_urls = set()
    pages_data = []
    total_urls_found = 1
    nodes_expanded = 0
    
    headers = {
        "User-Agent": "AletheiaCrawler/1.0 (+https://github.com/academic/aletheia; Research AI Assistant)"
    }
    
    while priority_queue and len(pages_data) < max_pages:
        # Pull the highest-priority page
        neg_priority, depth, current_url = heapq.heappop(priority_queue)
        
        # De-duplicate visited nodes
        if current_url in visited_urls:
            continue
        visited_urls.add(current_url)
        nodes_expanded += 1
        
        # Prevent searching beyond max designated depth
        if depth > max_depth:
            continue
            
        try:
            # Respectful, safe networking request with reasonable timeout
            response = requests.get(current_url, headers=headers, timeout=5, allow_redirects=True)
            
            # Ensure the resource is textual
            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" not in content_type and "text/plain" not in content_type:
                continue
                
            raw_html = response.text
            soup = BeautifulSoup(raw_html, "html.parser")
            
            # Store the raw text and HTML before deep cleaning
            pages_data.append({
                "url": current_url,
                "html": raw_html,
                "text": soup.get_text()
            })
            
            # Find and prioritize candidate internal child links
            for anchor in soup.find_all("a", href=True):
                href = anchor.get("href")
                
                # FIX: Verify that href exists and is a string to satisfy Pylance typing rules
                if not href or not isinstance(href, str):
                    continue
                
                # Resolve relative pathways to full absolute paths
                full_url = urllib.parse.urljoin(current_url, href)
                # Parse to verify if the link belongs strictly to the target host
                parsed_child = urllib.parse.urlparse(full_url)
                
                # Check for same domain crawling constraint to avoid runaway internet indexing
                if parsed_child.netloc == base_domain:
                    # Ignore common binary/media extensions to focus on pages
                    path_lower = parsed_child.path.lower()
                    if any(path_lower.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".gif", ".zip", ".mp4", ".mp3", ".css", ".js"]):
                        continue
                        
                    clean_child_url = urllib.parse.urlunparse((
                        parsed_child.scheme, parsed_child.netloc, parsed_child.path, parsed_child.params, parsed_child.query, ""
                    ))
                    
                    if clean_child_url not in visited_urls:
                        total_urls_found += 1
                        p_score = calculate_url_priority(clean_child_url)
                        # Push to the heuristic priority heap
                        heapq.heappush(priority_queue, (-p_score, depth + 1, clean_child_url))
                        
        except requests.exceptions.RequestException:
            # Fail-safe error-handling to prevent dropout on network timeouts or 404/500 errors
            # We log the event, skip the node, and continue extracting other pages
            continue
            
    execution_time = time.time() - start_time
    
    metrics = {
        "nodes_expanded": nodes_expanded,
        "execution_time_seconds": round(execution_time, 4),
        "total_urls_found": total_urls_found
    }
    
    return pages_data, metrics