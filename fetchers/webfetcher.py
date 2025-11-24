import os
import json
import time
import re
from googleapiclient.discovery import build
from urllib.parse import urlparse
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Simple domain scoring system (research-inspired but reliable)
DOMAIN_SCORES = {
    # High priority - authoritative sources
    'faa.gov': 100, 'nasa.gov': 95, 'icao.int': 90, 'easa.europa.eu': 85,
    'boeing.com': 80, 'airbus.com': 80, 'rolls-royce.com': 75, 'geaviation.com': 75,
    'aiaa.org': 70, 'aerospaceamerica.org': 70,
    
    # Medium priority - good technical content
    'wikipedia.org': 65, 'medium.com': 60, 'researchgate.net': 60,
    'sciencedirect.com': 60, 'ieee.org': 60,
    
    # Lower priority - user content (limit these)
    'reddit.com': 40, 'quora.com': 35, 'facebook.com': 30, 'twitter.com': 30,
    
    # Default score for unknown domains
    'default': 50
}

def get_domain_score(url):
    """Simple domain scoring based on authority (research-inspired)"""
    domain = urlparse(url).netloc.lower().replace('www.', '')
    
    # Check for exact matches
    for key_domain, score in DOMAIN_SCORES.items():
        if domain.endswith(key_domain):
            return score
    
    # Check for TLD-based scoring
    if domain.endswith('.gov') or domain.endswith('.mil'):
        return 90
    elif domain.endswith('.edu'):
        return 80
    elif domain.endswith('.org'):
        return 70
    elif domain.endswith('.com'):
        return 60
    
    return DOMAIN_SCORES['default']

def is_relevant_result(result, original_query):
    """Basic relevance filtering to avoid math/geometry results for aviation queries"""
    title = result.get('title', '').lower()
    snippet = result.get('snippet', '').lower()
    url = result.get('link', '').lower()
    
    # Always relevant if it contains these aerospace terms
    aerospace_terms = ['aircraft', 'airplane', 'aviation', 'aerospace', 'engine', 'jet', 'flight', 
                      'wing', 'thrust', 'fuselage', 'cockpit', 'pilot', 'aerodynamics']
    
    has_aerospace_context = any(term in title or term in snippet or term in url for term in aerospace_terms)
    
    # Block irrelevant terms
    irrelevant_terms = ['math', 'geometry', 'triangle', 'algebra', 'calculus', 'equation', 
                       'formula', 'theorem', 'proof', 'homework', 'textbook']
    
    has_irrelevant_context = any(term in title or term in snippet or term in url for term in irrelevant_terms)
    
    # Special handling for ambiguous terms
    if 'quad' in original_query.lower() and not ('engine' in original_query.lower() or 'aircraft' in original_query.lower()):
        if any(term in title+snippet+url for term in ['math', 'geometry', 'quadrilateral', 'bike', 'atv']):
            return False
    
    return has_aerospace_context and not has_irrelevant_context

def fetch_web_results(queries, session_dir, original_query):
    """Fixed web fetcher - reliable with research elements"""
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("SEARCH_ENGINE_ID")
    
    service = build("customsearch", "v1", developerKey=api_key)
    all_results = []
    
    # Track domains to avoid duplicates
    seen_domains = set()
    domain_counts = {}
    
    print("\n🔍 Searching web with domain-aware ranking...")
    
    for query in tqdm(queries, desc="Processing queries"):
        try:
            results = service.cse().list(
                q=query,
                cx=cse_id,
                num=5,  # Get more results to filter from
                safe='active'
            ).execute()
            
            items = results.get('items', [])
            
            for item in items:
                url = item['link']
                domain = urlparse(url).netloc.lower().replace('www.', '')
                
                # Skip if we've already seen this domain
                if domain in seen_domains and len(seen_domains) > 5:
                    continue
                
                # Skip low-quality domains
                if any(bad in domain for bad in ['porn', 'xxx', 'casino', 'gambling', 'dating']):
                    continue
                
                # Relevance filtering
                if not is_relevant_result(item, original_query):
                    continue
                
                # Get domain type for limiting purposes
                domain_type = 'other'
                if any(term in domain for term in ['reddit', 'facebook', 'twitter', 'instagram']):
                    domain_type = 'social'
                
                # Limit social media sources
                if domain_type == 'social':
                    domain_counts[domain_type] = domain_counts.get(domain_type, 0) + 1
                    if domain_counts[domain_type] > 2:  # Max 2 social media sources
                        continue
                
                # Calculate score and add to results
                score = get_domain_score(url)
                all_results.append({
                    "title": item['title'],
                    "link": item['link'],
                    "snippet": item['snippet'],
                    "query": query,
                    "domain_score": score,
                    "domain_type": domain_type
                })
                
                seen_domains.add(domain)
                if len(all_results) >= 15:  # Hard limit
                    break
            
            time.sleep(1)  # Rate limiting
            
            if len(all_results) >= 15:
                break
                
        except Exception as e:
            print(f"⚠️ Error searching for '{query}': {str(e)}")
    
    # Sort by domain score
    all_results.sort(key=lambda x: x['domain_score'], reverse=True)
    
    # Save results
    results_data = {
        "original_query": original_query,
        "processed_queries": queries,
        "results": all_results[:10]  # Only keep top 10 results
    }
    
    with open(os.path.join(session_dir, "queries/web_results.json"), "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n✅ Found {len(all_results)} relevant web results (displaying top 10)")
    for result in all_results[:10]:
        print(f"   • {result['title'][:50]}... ({result['domain_score']}/100)")
    
    return all_results[:10]
























# import os
# import json
# import time
# from googleapiclient.discovery import build
# from dotenv import load_dotenv
# from tqdm import tqdm

# load_dotenv()

# def fetch_web_results(queries, session_dir):
#     api_key = os.getenv("GOOGLE_API_KEY")
#     cse_id = os.getenv("SEARCH_ENGINE_ID")
    
#     service = build("customsearch", "v1", developerKey=api_key)
#     all_results = []
    
#     for query in tqdm(queries, desc="Processing queries"):
#         try:
#             results = service.cse().list(
#                 q=query,
#                 cx=cse_id,
#                 num=3  # Max 3 results per query
#             ).execute()
            
#             items = results.get('items', [])
#             for item in items:
#                 all_results.append({
#                     "title": item['title'],
#                     "link": item['link'],
#                     "snippet": item['snippet'],
#                     "query": query
#                 })
            
#             time.sleep(1)  # Respect rate limits
            
#         except Exception as e:
#             print(f"Error searching for '{query}': {str(e)}")
    
#     # Save results
#     with open(os.path.join(session_dir, "queries/web_results.json"), "w", encoding="utf-8") as f:
#         json.dump(all_results, f, indent=2)
    
#     return all_results