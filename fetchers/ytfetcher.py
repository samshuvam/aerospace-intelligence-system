import os
import json
import time
from googleapiclient.discovery import build
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

def fetch_youtube_results(queries, session_dir):
    api_key = os.getenv("YOUTUBE_API_KEY")
    youtube = build("youtube", "v3", developerKey=api_key)
    all_results = []
    
    for query in tqdm(queries, desc="Searching YouTube"):
        try:
            search_response = youtube.search().list(
                q=f"aerospace engineering {query}",
                part="snippet",
                maxResults=2, 
                type="video",
                relevanceLanguage="en",
                safeSearch="strict"
            ).execute()
            
            for item in search_response.get("items", []):
                video_id = item["id"]["videoId"]
                title = item["snippet"]["title"]
                all_results.append({
                    "video_id": video_id,
                    "title": title,
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "query": query
                })
            
            time.sleep(1)
            
        except Exception as e:
            print(f"⚠️ YouTube error for '{query}': {str(e)}")
    
    # Save results
    with open(os.path.join(session_dir, "queries/yt_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    
    return all_results