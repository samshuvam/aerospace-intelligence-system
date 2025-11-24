import os
import json
import time
from newspaper import Article
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from tqdm import tqdm
from urllib.parse import urlparse

def safe_extract(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text[:5000]
    except:
        return None

def selenium_extract(url):
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    driver = None
    try:
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=chrome_options
        )
        driver.set_page_load_timeout(15)
        driver.get(url)
        time.sleep(2)
        
        paragraphs = driver.find_elements("tag name", "p")
        content = "\n".join([p.text for p in paragraphs if len(p.text) > 80])
        return content[:5000]
    except Exception as e:
        print(f".Selenium error for {url}: {str(e)}")
        return None
    finally:
        if driver:
            driver.quit()

def extract_web_content(results, session_dir):
    extracted_dir = os.path.join(session_dir, "extracted/web")
    os.makedirs(extracted_dir, exist_ok=True)
    
    for i, result in enumerate(tqdm(results, desc="Extracting web content")):
        url = result["link"]
        domain = urlparse(url).netloc.replace("www.", "")
        filename = f"web_{i}_{domain[:10]}.txt"
        filepath = os.path.join(extracted_dir, filename)
        
        # Skip if already extracted
        if os.path.exists(filepath):
            continue
        
        content = safe_extract(url) or selenium_extract(url)
        
        if content and len(content) > 100:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"SOURCE: {url}\n")
                f.write(f"QUERY: {result['query']}\n")
                f.write("="*50 + "\n\n")
                f.write(content)
        else:
            print(f"Failed to extract content from {url}")
        
        time.sleep(0.5)