import os
import time
from pytubefix import YouTube
from pytubefix.exceptions import PytubeFixError, VideoUnavailable
import whisper
from tqdm import tqdm
import logging

# Suppress unnecessary logging
logging.getLogger("pytubefix").setLevel(logging.ERROR)
logging.getLogger("whisper").setLevel(logging.ERROR)

def download_audio(video_id, temp_dir):
    """Download audio using pytubefix (NO yt-dlp)"""
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        print(f"  → Downloading audio from: {url}")
        
        yt = YouTube(url, use_oauth=False, allow_oauth_cache=False)
        
        audio_stream = yt.streams.filter(only_audio=True).order_by('abr').desc().first()
        
        if not audio_stream:
            print(f"  ✗ No audio streams available for {video_id}")
            return None
        
        output_path = audio_stream.download(
            output_path=temp_dir,
            filename=f"{video_id}.mp4"
        )
        
        print(f"  Audio downloaded successfully: {os.path.basename(output_path)}")
        return output_path
        
    except (PytubeFixError, VideoUnavailable) as e:
        print(f"  Pytubefix error: {str(e)}")
        return None
    except Exception as e:
        print(f"  Unexpected download error: {str(e)}")
        return None

def transcribe_audio(audio_path, video_id):
    try:
        print("  Transcribing with Whisper")
        
        # Load Whisper model (base is fast, small is more accurate)
        model = whisper.load_model("base")
        
        # Transcribe audio
        result = model.transcribe(
            audio_path,
            fp16=False,
            language="en",
            verbose=False
        )
        
        transcript = result["text"]
        print(f"  ✓ Transcription complete: {len(transcript)} characters")
        return transcript
        
    except Exception as e:
        print(f"  ✗ Whisper transcription failed: {str(e)}")
        return None

def extract_youtube_content(results, session_dir):
    """SIMPLIFIED YOUTUBE EXTRACTION PIPELINE (NO yt-dlp, NO Transcript API)"""
    print("\n" + "="*60)
    print("YOUTUBE CONTENT EXTRACTION (DIRECT AUDIO + WHISPER ONLY)")
    print("="*60)
    
    extracted_dir = os.path.join(session_dir, "extracted/youtube")
    temp_dir = os.path.join(session_dir, "temp")
    os.makedirs(extracted_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    # Get unique videos only
    MAX_VIDEOS = 10
    unique_videos = {}
    for result in results:
        if len(unique_videos) >= MAX_VIDEOS:
            break
        vid = result["video_id"]
        if vid not in unique_videos:
            unique_videos[vid] = result
    
    print(f"Found {len(unique_videos)} unique videos to process")
    successful = 0
    
    for i, (video_id, result) in enumerate(unique_videos.items()):
        print(f"\n▶️ PROCESSING VIDEO {i+1}/{len(unique_videos)}")
        print(f"Title: {result['title']}")
        print(f"URL: https://www.youtube.com/watch?v={video_id}")
        
        # Skip if already processed
        output_file = os.path.join(extracted_dir, f"yt_{i}_{video_id}.txt")
        if os.path.exists(output_file):
            print(f"  ✓ Already processed. Skipping.")
            successful += 1
            continue
        
        try:
            # STEP 1: Download audio using pytubefix (NO yt-dlp)
            audio_path = download_audio(video_id, temp_dir)
            
            if not audio_path or not os.path.exists(audio_path):
                print(f"  ✗ Failed to download audio for {video_id}")
                continue
            
            # STEP 2: Transcribe with Whisper (NO YouTube Transcript API)
            transcript = transcribe_audio(audio_path, video_id)
            
            # STEP 3: Clean up audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)
                print(f"  → Temporary audio file cleaned up")
            
            # STEP 4: Save transcript if successful
            if transcript and len(transcript.strip()) > 500:
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(f"VIDEO TITLE: {result['title']}\n")
                    f.write(f"URL: https://www.youtube.com/watch?v={video_id}\n")
                    f.write(f"QUERY: {result['query']}\n")
                    f.write("="*60 + "\n\n")
                    f.write(transcript.strip())
                
                print(f"  ✓ Successfully saved transcript to: {os.path.basename(output_file)}")
                successful += 1
            else:
                print(f"  ✗ Transcription too short or failed for {video_id}")
        
        except Exception as e:
            print(f"  ✗ Critical error processing {video_id}: {str(e)}")
        
        # Rate limiting to avoid YouTube blocks
        if i < len(unique_videos) - 1:
            print("  → Pausing 2 seconds before next video (rate limiting)")
            time.sleep(2)
    
    # Final cleanup
    if os.path.exists(temp_dir):
        try:
            for file in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            if not os.listdir(temp_dir):  # Only remove if empty
                os.rmdir(temp_dir)
        except Exception as e:
            print(f"  → Cleanup warning: {str(e)}")
    
    print(f"\n" + "="*60)
    print(f"YOUTUBE EXTRACTION COMPLETE: {successful}/{len(unique_videos)} videos processed")
    print("="*60)
    
    return successful > 0













# import os
# import time
# import requests
# import whisper
# from pydub import AudioSegment
# from tqdm import tqdm
# import logging
# from urllib.parse import urlparse, parse_qs
# import json

# logging.getLogger("pytube").setLevel(logging.ERROR)

# def download_youtube_audio(video_id, temp_dir):
#     """Direct audio download using yt-dlp with minimal dependencies"""
#     try:
#         import yt_dlp
        
#         ydl_opts = {
#             'format': 'bestaudio/best',
#             'postprocessors': [{
#                 'key': 'FFmpegExtractAudio',
#                 'preferredcodec': 'mp3',
#                 'preferredquality': '128',
#             }],
#             'outtmpl': os.path.join(temp_dir, f'{video_id}.%(ext)s'),
#             'quiet': True,
#             'noplaylist': True,
#             'extractor_args': {'youtube': {'player_client': ['ios']}},
#             'http_headers': {
#                 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
#             }
#         }
        
#         with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#             ydl.download([f'https://www.youtube.com/watch?v={video_id}'])
        
#         audio_path = os.path.join(temp_dir, f'{video_id}.mp3')
#         return audio_path if os.path.exists(audio_path) else None
        
#     except Exception as e:
#         print(f"  → Direct download failed: {str(e)}")
#         return None

# def transcribe_audio(audio_path):
#     """Transcribe audio using Whisper with progress display"""
#     try:
#         print("  → Transcribing audio with Whisper (this may take 1-2 minutes)...")
#         model = whisper.load_model("base")
#         result = model.transcribe(audio_path, fp16=False, verbose=False)
#         return result["text"]
#     except Exception as e:
#         print(f"  → Transcription error: {str(e)}")
#         return None

# def extract_youtube_content(results, session_dir):
#     """Simplified YouTube processing focused only on audio extraction"""
#     print("\n EXTRACTING YOUTUBE AUDIO CONTENT (SIMPLIFIED APPROACH)")
    
#     extracted_dir = os.path.join(session_dir, "extracted/youtube")
#     temp_dir = os.path.join(session_dir, "temp")
#     os.makedirs(extracted_dir, exist_ok=True)
#     os.makedirs(temp_dir, exist_ok=True)
    
#     # Get unique videos only
#     unique_videos = {}
#     for result in results:
#         vid = result["video_id"]
#         if vid not in unique_videos:
#             unique_videos[vid] = result
    
#     successful_extractions = 0
#     total_videos = len(unique_videos)
    
#     for i, (video_id, result) in enumerate(unique_videos.items()):
#         print(f"\n⏯  Processing video {i+1}/{total_videos}: {result['title'][:50]}...")
        
#         filename = f"yt_{i}_{video_id}.txt"
#         filepath = os.path.join(extracted_dir, filename)
        
#         # Skip if already processed
#         if os.path.exists(filepath):
#             print(f"   Already processed, skipping")
#             successful_extractions += 1
#             continue
        
#         try:
#             # STEP 1: Download audio directly
#             print("  → Downloading audio...")
#             audio_path = download_youtube_audio(video_id, temp_dir)
            
#             if not audio_path or not os.path.exists(audio_path):
#                 print(f"   Failed to download audio for {video_id}")
#                 continue
            
#             # STEP 2: Transcribe audio
#             transcript = transcribe_audio(audio_path)
            
#             # STEP 3: Clean up audio file
#             if os.path.exists(audio_path):
#                 os.remove(audio_path)
            
#             # STEP 4: Save transcript if successful
#             if transcript and len(transcript.strip()) > 300:
#                 with open(filepath, "w", encoding="utf-8") as f:
#                     f.write(f"VIDEO TITLE: {result['title']}\n")
#                     f.write(f"URL: https://www.youtube.com/watch?v={video_id}\n")
#                     f.write(f"QUERY: {result['query']}\n")
#                     f.write("="*50 + "\n\n")
#                     # Limit to 12000 characters for LLM context
#                     f.write(transcript[:12000])
                
#                 print(f"   Successfully transcribed {len(transcript)} characters")
#                 successful_extractions += 1
#             else:
#                 print(f"   Transcription too short or failed")
        
#         except Exception as e:
#             print(f"   Error processing {video_id}: {str(e)}")
        
#         time.sleep(1)  # Rate limiting
    
#     # Cleanup temp directory
#     if os.path.exists(temp_dir):
#         try:
#             for file in os.listdir(temp_dir):
#                 file_path = os.path.join(temp_dir, file)
#                 if os.path.isfile(file_path):
#                     os.remove(file_path)
#             os.rmdir(temp_dir)
#         except Exception as e:
#             print(f"  → Cleanup warning: {str(e)}")
    
#     print(f"\n YouTube processing complete: {successful_extractions}/{total_videos} videos successfully processed")














# import os
# import json
# import time
# import requests
# from pytube import YouTube
# from pytube.exceptions import PytubeError
# from youtube_transcript_api import YouTubeTranscriptApi
# from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
# import whisper
# from tqdm import tqdm
# import logging

# # Set up logging to suppress pytube noise
# logging.getLogger("pytube").setLevel(logging.ERROR)

# def get_transcript(video_id):
#     """Try to get transcript using YouTube Transcript API"""
#     try:
#         # Try to get English transcript first
#         transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
#         return " ".join([entry['text'] for entry in transcript])
#     except (TranscriptsDisabled, NoTranscriptFound):
#         try:
#             # Try to get any available transcript and translate to English
#             transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
#             for transcript in transcript_list:
#                 if transcript.language_code != 'en':
#                     translated_transcript = transcript.translate('en')
#                     data = translated_transcript.fetch()
#                     return " ".join([entry['text'] for entry in data])
#         except:
#             return None
#     except Exception as e:
#         print(f"Transcript API error: {str(e)}")
#         return None

# def download_audio(video_url, output_path):
#     """Download audio using PyTube instead of yt-dlp"""
#     try:
#         yt = YouTube(video_url)
#         audio_stream = yt.streams.filter(only_audio=True).first()
        
#         if not audio_stream:
#             print(f"No audio stream found for {video_url}")
#             return False
        
#         # Download to temporary location then rename
#         temp_path = audio_stream.download(output_path=os.path.dirname(output_path))
#         os.rename(temp_path, output_path)
#         return True
#     except PytubeError as e:
#         print(f"PyTube error: {str(e)}")
#         return False
#     except Exception as e:
#         print(f"Download error: {str(e)}")
#         return False

# def transcribe_audio(audio_path):
#     """Transcribe audio using Whisper"""
#     try:
#         model = whisper.load_model("base")
#         result = model.transcribe(audio_path, fp16=False)
#         return result["text"]
#     except Exception as e:
#         print(f"Whisper transcription error: {str(e)}")
#         return None

# def extract_youtube_content(results, session_dir):
#     """Process YouTube videos with transcript-first approach"""
#     extracted_dir = os.path.join(session_dir, "extracted/youtube")
#     temp_dir = os.path.join(session_dir, "temp")
#     os.makedirs(extracted_dir, exist_ok=True)
#     os.makedirs(temp_dir, exist_ok=True)
    
#     # Only process unique videos (avoid duplicates)
#     unique_videos = {}
#     for result in results:
#         vid = result["video_id"]
#         if vid not in unique_videos:
#             unique_videos[vid] = result
    
#     for i, (video_id, result) in enumerate(tqdm(unique_videos.items(), desc="Processing YouTube videos")):
#         filename = f"yt_{i}_{video_id}.txt"
#         filepath = os.path.join(extracted_dir, filename)
        
#         # Skip if already processed
#         if os.path.exists(filepath):
#             print(f"✓ Skipping already processed video: {video_id}")
#             continue
        
#         print(f"\n🎬 Processing: {result['title']}")
        
#         try:
#             # STEP 1: Try to get transcript first (fastest method)
#             print("  → Attempting to get transcript...")
#             transcript = get_transcript(video_id)
            
#             # STEP 2: If transcript fails, download audio and transcribe
#             if not transcript or len(transcript.strip()) < 200:
#                 print("  → Transcript unavailable or too short, falling back to Whisper...")
                
#                 # Download audio
#                 audio_path = os.path.join(temp_dir, f"{video_id}.mp3")
#                 success = download_audio(result["url"], audio_path)
                
#                 if success and os.path.exists(audio_path):
#                     # Transcribe audio
#                     transcript = transcribe_audio(audio_path)
#                     os.remove(audio_path)
#                 else:
#                     print(f"  → Failed to download audio for {video_id}")
#                     continue
            
#             # STEP 3: Save successful transcript
#             if transcript and len(transcript.strip()) > 300:
#                 with open(filepath, "w", encoding="utf-8") as f:
#                     f.write(f"VIDEO TITLE: {result['title']}\n")
#                     f.write(f"URL: {result['url']}\n")
#                     f.write(f"QUERY: {result['query']}\n")
#                     f.write("="*50 + "\n\n")
#                     # Limit to 10k characters to avoid overwhelming the LLM
#                     f.write(transcript[:10000])
#                 print(f"  ✓ Successfully processed {video_id}")
#             else:
#                 print(f"  → Insufficient content from {video_id}")
        
#         except Exception as e:
#             print(f"⚠️ Error processing {video_id}: {str(e)}")
        
#         time.sleep(1)  # Rate limiting
    
#     # Cleanup temp directory
#     if os.path.exists(temp_dir):
#         try:
#             for file in os.listdir(temp_dir):
#                 file_path = os.path.join(temp_dir, file)
#                 if os.path.isfile(file_path):
#                     os.remove(file_path)
#             os.rmdir(temp_dir)
#         except:
#             pass