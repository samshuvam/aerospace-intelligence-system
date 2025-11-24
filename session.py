import os
import json
import time
from queryprocess import process_query
from fetchers.webfetcher import fetch_web_results
from fetchers.ytfetcher import fetch_youtube_results
from extractors.webextractor import extract_web_content
from extractors.ytextractor import extract_youtube_content
from merge import merge_files
from llm import generate_answer

def initialize_session():
    os.makedirs("data", exist_ok=True)
    session_dirs = [d for d in os.listdir("data") if d.startswith("session_")]
    session_num = len(session_dirs) + 1
    session_dir = os.path.join("data", f"session_{session_num}")
    os.makedirs(session_dir, exist_ok=True)
    
    # Create subdirectories
    for subdir in ["queries", "extracted/web", "extracted/youtube", "merged"]:
        os.makedirs(os.path.join(session_dir, subdir), exist_ok=True)
    
    return session_dir, session_num

def save_query(session_dir, original_query, processed_queries):
    with open(os.path.join(session_dir, "queries/original.txt"), "w", encoding="utf-8") as f:
        f.write(original_query)
    
    with open(os.path.join(session_dir, "queries/processed.json"), "w", encoding="utf-8") as f:
        json.dump(processed_queries, f, indent=2)

def main():
    print("="*50)
    print("AEROSPACE INTELLIGENCE SYSTEM")
    print("="*50)
    
    while True:
        print("\n1. Start new session")
        print("2. Exit system")
        choice = input("\nSelect option: ").strip()
        
        if choice == "1":
            session_dir, session_num = initialize_session()
            print(f"\n Session {session_num} started!")
            
            while True:
                query = input("\nEnter your aerospace query (or 'exit' to end session): ").strip()
                if query.lower() == "exit":
                    break
                
                print("\n Processing query through NLP pipeline...")
                processed_queries = process_query(query)
                save_query(session_dir, query, processed_queries)
                
                print("\n Fetching web resources...")
                web_results = fetch_web_results(processed_queries, session_dir, query)
                
                print("\n Fetching YouTube resources...")
                yt_results = fetch_youtube_results(processed_queries, session_dir)
                
                print("\n Extracting web content...")
                extract_web_content(web_results, session_dir)
                
                print("\n Extracting YouTube content...")
                extract_youtube_content(yt_results, session_dir)
                
                print("\n Merging all content...")
                if not web_results and not yt_results:
                    print("❌ ERROR: No content was successfully extracted from web or YouTube sources.")
                    print("   Please try a different query or check your internet connection.")
                    continue
    
                merged_path = merge_files(session_dir)
                
                print("\n Generating final answer with Mistral-7B...")
                final_answer = generate_answer(merged_path, query)
                
                print("\n" + "="*50)
                print("FINAL ANSWER")
                print("="*50)
                print(final_answer)
                print("="*50)
                
                # Save final answer
                with open(os.path.join(session_dir, "final_answer.txt"), "w", encoding="utf-8") as f:
                    f.write(final_answer)
            
            print(f"\n Session {session_num} completed. Data saved to: {session_dir}")
        
        elif choice == "2":
            print("\n Thank you for using the Aerospace Intelligence System!")
            break
        else:
            print("\n Invalid option. Please try again.")

if __name__ == "__main__":
    main()