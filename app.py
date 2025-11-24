import streamlit as st
import os
import json
import time
import sys
from datetime import datetime

# Import your existing modules
try:
    from queryprocess import process_query
    from fetchers.webfetcher import fetch_web_results
    from fetchers.ytfetcher import fetch_youtube_results
    from extractors.webextractor import extract_web_content
    from extractors.ytextractor import extract_youtube_content
    from merge import merge_files
    from llm import generate_answer
except ImportError as e:
    st.error(f"Error importing modules: {str(e)}")
    st.stop()

# ====================== SESSION STATE INITIALIZATION ======================
# This MUST be at the very top, before any UI elements
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.messages = []
    st.session_state.session_dir = None
    st.session_state.session_num = 1
    st.session_state.processing = False
    st.session_state.logs = []
    st.session_state.current_query = None
    st.session_state.final_answer = None
    st.session_state.processing_step = "idle"
    st.session_state.error_message = None
    print("✅ Session state properly initialized")

# ====================== HELPER FUNCTIONS ======================
def initialize_session():
    """Create a new session directory structure"""
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
    """Save query information to session directory"""
    with open(os.path.join(session_dir, "queries/original.txt"), "w", encoding="utf-8") as f:
        f.write(original_query)
    
    with open(os.path.join(session_dir, "queries/processed.json"), "w", encoding="utf-8") as f:
        json.dump(processed_queries, f, indent=2)

def log_step(step_name, message):
    """Add a log entry with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{timestamp}] {step_name}: {message}")
    # Also print to console for debugging
    print(f"[{timestamp}] {step_name}: {message}")

def reset_processing_state():
    """Reset all processing-related states"""
    st.session_state.processing = False
    st.session_state.logs = []
    st.session_state.processing_step = "idle"
    st.session_state.error_message = None
    st.session_state.final_answer = None

# ====================== MAIN PROCESSING FUNCTION ======================
def process_query_sequentially(query):
    """Process a query through the entire pipeline (synchronous version)"""
    try:
        # Initialize session if needed
        if not st.session_state.session_dir:
            st.session_state.session_dir, st.session_state.session_num = initialize_session()
        
        # Reset processing state
        reset_processing_state()
        st.session_state.processing = True
        st.session_state.current_query = query
        st.session_state.processing_step = "starting"
        
        log_step("🔄", f"Starting new query: '{query}'")
        
        # STEP 1: Query Processing
        st.session_state.processing_step = "query_processing"
        log_step("🧠", "Processing query through NLP pipeline...")
        processed_queries = process_query(query)
        save_query(st.session_state.session_dir, query, processed_queries)
        log_step("✅", f"Generated {len(processed_queries)} processed queries")
        
        # STEP 2: Web Search
        st.session_state.processing_step = "web_search"
        log_step("🔍", "Fetching web resources...")
        web_results = fetch_web_results(processed_queries, st.session_state.session_dir, query)
        log_step("✅", f"Found {len(web_results)} web results")
        
        # STEP 3: YouTube Search
        st.session_state.processing_step = "youtube_search"
        log_step("📺", "Fetching YouTube resources...")
        yt_results = fetch_youtube_results(processed_queries, st.session_state.session_dir)
        log_step("✅", f"Found {len(yt_results)} YouTube videos")
        
        # STEP 4: Web Content Extraction
        st.session_state.processing_step = "web_extraction"
        log_step("🌐", "Extracting web content...")
        extract_web_content(web_results, st.session_state.session_dir)
        log_step("✅", "Web content extraction completed")
        
        # STEP 5: YouTube Content Extraction
        st.session_state.processing_step = "youtube_extraction"
        log_step("🔊", "Extracting YouTube content...")
        extract_youtube_content(yt_results, st.session_state.session_dir)
        log_step("✅", "YouTube content extraction completed")
        
        # STEP 6: Content Merging
        st.session_state.processing_step = "merging"
        log_step("🔀", "Merging all content...")
        merged_path = merge_files(st.session_state.session_dir)
        log_step("✅", "Content merging completed")
        
        # STEP 7: LLM Answer Generation
        st.session_state.processing_step = "llm_generation"
        log_step("🤖", "Generating final answer with Mistral-7B...")
        final_answer = generate_answer(merged_path, query)
        st.session_state.final_answer = final_answer
        log_step("✅", "Answer generation completed")
        
        # STEP 8: Save and Add to Chat
        with open(os.path.join(st.session_state.session_dir, "final_answer.txt"), "w", encoding="utf-8") as f:
            f.write(final_answer)
        
        st.session_state.messages.append({
            "role": "user",
            "content": query
        })
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": final_answer,
            "session_dir": st.session_state.session_dir
        })
        
        log_step("🎉", "Session completed successfully!")
        st.session_state.processing_step = "completed"
        
    except Exception as e:
        error_msg = f"❌ ERROR: {str(e)}"
        log_step("❌", error_msg)
        st.session_state.error_message = str(e)
        st.session_state.processing_step = "error"
    finally:
        st.session_state.processing = False

# ====================== UI LAYOUT ======================
st.set_page_config(
    page_title="Aerospace Intelligence System",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.chat-message {
    padding: 1.5rem; 
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
}
.chat-message.user {
    background-color: #2b3540;
}
.chat-message.bot {
    background-color: #1a2430;
}
.final-answer {
    background-color: #1a2430;
    border-left: 4px solid #1f77b4;
    padding: 1.5rem;
    margin: 1.5rem 0;
    border-radius: 0.3rem;
}
.log-container {
    background-color: #0e1117;
    border: 1px solid #4a4a4a;
    border-radius: 0.5rem;
    padding: 1rem;
    margin: 1rem 0;
    font-family: monospace;
    font-size: 0.9rem;
    max-height: 400px;
    overflow-y: auto;
    white-space: pre-wrap;
}
.session-info {
    background-color: #1a2430;
    padding: 0.8rem;
    border-radius: 0.3rem;
    margin-bottom: 1rem;
    font-size: 0.9rem;
}
.step-container {
    display: flex;
    align-items: center;
    margin: 0.5rem 0;
}
.step-icon {
    font-size: 1.2rem;
    margin-right: 0.5rem;
    min-width: 2rem;
}
.step-text {
    flex-grow: 1;
}
.step-pending {
    color: #a0a0a0;
}
.step-active {
    color: #1f77b4;
    font-weight: bold;
}
.step-complete {
    color: #2ca02c;
}
.step-error {
    color: #d62728;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🚀 Control Panel")
    
    # Session controls
    st.subheader("Session Management")
    if st.button("🆕 New Session", use_container_width=True):
        st.session_state.session_dir, st.session_state.session_num = initialize_session()
        st.session_state.messages = []
        st.session_state.logs = []
        reset_processing_state()
        st.success(f"✅ Created session {st.session_state.session_num}")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.logs = []
        reset_processing_state()
        st.success("✅ Chat cleared")
    
    # Session info
    st.subheader("Session Info")
    if st.session_state.session_dir:
        st.markdown(f'<div class="session-info">Session: {st.session_state.session_num}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="session-info">Directory: {st.session_state.session_dir}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="session-info">No active session</div>', unsafe_allow_html=True)
    
    # System info
    st.subheader("System Info")
    st.markdown(f'<div class="session-info">Python: {sys.version.split()[0]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="session-info">Status: {"🚀 Running" if not st.session_state.processing else "🟡 Processing"}</div>', unsafe_allow_html=True)

# Main content
st.title("✈️ Aerospace Intelligence System")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown("### 🤖 Aerospace Intelligence System")
            st.markdown('<div class="final-answer">', unsafe_allow_html=True)
            st.markdown(message["content"])
            st.markdown('</div>', unsafe_allow_html=True)
            
            if "session_dir" in message:
                session_path = os.path.relpath(message["session_dir"], start=os.getcwd())
                st.markdown(f"📁 *Session data saved to: `{session_path}`*")

# Processing status and logs
if st.session_state.processing or st.session_state.logs:
    st.subheader("📋 Processing Status")
    
    # Step-by-step progress
    steps = [
        ("starting", "🚀 Starting"),
        ("query_processing", "🧠 Query Processing"),
        ("web_search", "🔍 Web Search"),
        ("youtube_search", "📺 YouTube Search"),
        ("web_extraction", "🌐 Web Extraction"),
        ("youtube_extraction", "🔊 YouTube Extraction"),
        ("merging", "🔀 Content Merging"),
        ("llm_generation", "🤖 Answer Generation"),
        ("completed", "✅ Completed")
    ]

    # ---------- SAFE INDEX LOOKUP ----------
    def get_step_index(step_id):
        for i, (sid, _) in enumerate(steps):
            if sid == step_id:
                return i
        return -1  # not found → treat as before the first step

    current_step_index = get_step_index(st.session_state.processing_step)
    # ----------------------------------------

    for step_id, step_name in steps:
        this_index = get_step_index(step_id)

        if st.session_state.processing_step == "error" and step_id == "error":
            step_class = "step-error"

        elif step_id == st.session_state.processing_step:
            step_class = "step-active"

        elif this_index != -1 and this_index < current_step_index:
            step_class = "step-complete"

        elif st.session_state.processing_step == "completed":
            step_class = "step-complete"

        else:
            step_class = "step-pending"

        st.markdown(f"""
        <div class="step-container">
            <span class="step-icon {step_class}">•</span>
            <span class="step-text {step_class}">{step_name}</span>
        </div>
        """, unsafe_allow_html=True)

    
    # Display logs
    if st.session_state.logs:
        log_content = "\n".join(st.session_state.logs)
        st.markdown(f'<div class="log-container">{log_content}</div>', unsafe_allow_html=True)
    
    # Stop button during processing
    if st.session_state.processing:
        if st.button("⏹️ Stop Processing", key="stop_button"):
            st.warning("🛑 Processing interrupted by user")
            reset_processing_state()
            st.rerun()

# Chat input
if not st.session_state.processing:
    user_query = st.chat_input("Enter your aerospace query (e.g., 'why do airplanes have two engines')", key="user_input")
    
    if user_query and user_query.strip():
        # Start processing
        process_query_sequentially(user_query.strip())
        st.rerun()

# Welcome message for empty state
if not st.session_state.messages and not st.session_state.processing:
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background-color: #1a2430; border-radius: 0.5rem; margin: 1rem 0;">
        <h3>🚀 Welcome to the Aerospace Intelligence System</h3>
        <p style="font-size: 1.1rem; color: #a0a0a0;">
            This advanced information retrieval system combines multi-source data gathering with LLM-powered knowledge synthesis to answer complex aerospace engineering questions.
        </p>
        <p style="font-size: 1rem; color: #a0a0a0;">
            To get started, click the "New Session" button in the sidebar or simply type your query below.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### 💡 Example Queries:
    - Why do airplanes have two engines?
    - How do jet engines work?
    - What is ETOPS certification?
    - How do wings generate lift?
    - What are the advantages of composite materials in aircraft?
    """)

# Footer
st.markdown("""
<hr style="border: 1px solid #4a4a4a; margin: 2rem 0;">
<div style="text-align: center; color: #a0a0a0; font-size: 0.9rem;">
    Aerospace Intelligence System &copy; 2024 | Research-Enhanced Information Retrieval
</div>
""", unsafe_allow_html=True)

# Auto-rerun check for background processing
if st.session_state.processing:
    time.sleep(0.5)
    st.rerun()


















# import streamlit as st
# import os
# import json
# import time
# import sys
# from datetime import datetime

# # Import your existing modules
# try:
#     from queryprocess import process_query
#     from fetchers.webfetcher import fetch_web_results
#     from fetchers.ytfetcher import fetch_youtube_results
#     from extractors.webextractor import extract_web_content
#     from extractors.ytextractor import extract_youtube_content
#     from merge import merge_files
#     from llm import generate_answer
# except ImportError as e:
#     st.error(f"Error importing modules: {str(e)}")
#     st.stop()

# # ====================== SESSION STATE INITIALIZATION ======================
# # This MUST be at the very top, before any UI elements
# if 'initialized' not in st.session_state:
#     st.session_state.initialized = True
#     st.session_state.messages = []
#     st.session_state.session_dir = None
#     st.session_state.session_num = 1
#     st.session_state.processing = False
#     st.session_state.logs = []
#     st.session_state.current_query = None
#     st.session_state.final_answer = None
#     st.session_state.processing_step = "idle"
#     st.session_state.error_message = None
#     print("✅ Session state properly initialized")

# # ====================== HELPER FUNCTIONS ======================
# def initialize_session():
#     """Create a new session directory structure"""
#     os.makedirs("data", exist_ok=True)
#     session_dirs = [d for d in os.listdir("data") if d.startswith("session_")]
#     session_num = len(session_dirs) + 1
#     session_dir = os.path.join("data", f"session_{session_num}")
#     os.makedirs(session_dir, exist_ok=True)
    
#     # Create subdirectories
#     for subdir in ["queries", "extracted/web", "extracted/youtube", "merged"]:
#         os.makedirs(os.path.join(session_dir, subdir), exist_ok=True)
    
#     return session_dir, session_num

# def save_query(session_dir, original_query, processed_queries):
#     """Save query information to session directory"""
#     with open(os.path.join(session_dir, "queries/original.txt"), "w", encoding="utf-8") as f:
#         f.write(original_query)
    
#     with open(os.path.join(session_dir, "queries/processed.json"), "w", encoding="utf-8") as f:
#         json.dump(processed_queries, f, indent=2)

# def log_step(step_name, message):
#     """Add a log entry with timestamp"""
#     timestamp = datetime.now().strftime("%H:%M:%S")
#     st.session_state.logs.append(f"[{timestamp}] {step_name}: {message}")
#     # Also print to console for debugging
#     print(f"[{timestamp}] {step_name}: {message}")

# def reset_processing_state():
#     """Reset all processing-related states"""
#     st.session_state.processing = False
#     st.session_state.logs = []
#     st.session_state.processing_step = "idle"
#     st.session_state.error_message = None
#     st.session_state.final_answer = None

# # ====================== MAIN PROCESSING FUNCTION ======================
# def process_query_sequentially(query):
#     """Process a query through the entire pipeline (synchronous version)"""
#     try:
#         # Initialize session if needed
#         if not st.session_state.session_dir:
#             st.session_state.session_dir, st.session_state.session_num = initialize_session()
        
#         # Reset processing state
#         reset_processing_state()
#         st.session_state.processing = True
#         st.session_state.current_query = query
#         st.session_state.processing_step = "starting"
        
#         log_step("🔄", f"Starting new query: '{query}'")
        
#         # STEP 1: Query Processing
#         st.session_state.processing_step = "query_processing"
#         log_step("🧠", "Processing query through NLP pipeline...")
#         processed_queries = process_query(query)
#         save_query(st.session_state.session_dir, query, processed_queries)
#         log_step("✅", f"Generated {len(processed_queries)} processed queries")
        
#         # STEP 2: Web Search
#         st.session_state.processing_step = "web_search"
#         log_step("🔍", "Fetching web resources...")
#         web_results = fetch_web_results(processed_queries, st.session_state.session_dir, query)
#         log_step("✅", f"Found {len(web_results)} web results")
        
#         # STEP 3: YouTube Search
#         st.session_state.processing_step = "youtube_search"
#         log_step("📺", "Fetching YouTube resources...")
#         yt_results = fetch_youtube_results(processed_queries, st.session_state.session_dir)
#         log_step("✅", f"Found {len(yt_results)} YouTube videos")
        
#         # STEP 4: Web Content Extraction
#         st.session_state.processing_step = "web_extraction"
#         log_step("🌐", "Extracting web content...")
#         extract_web_content(web_results, st.session_state.session_dir)
#         log_step("✅", "Web content extraction completed")
        
#         # STEP 5: YouTube Content Extraction
#         st.session_state.processing_step = "youtube_extraction"
#         log_step("🔊", "Extracting YouTube content...")
#         extract_youtube_content(yt_results, st.session_state.session_dir)
#         log_step("✅", "YouTube content extraction completed")
        
#         # STEP 6: Content Merging
#         st.session_state.processing_step = "merging"
#         log_step("🔀", "Merging all content...")
#         merged_path = merge_files(st.session_state.session_dir)
#         log_step("✅", "Content merging completed")
        
#         # STEP 7: LLM Answer Generation
#         st.session_state.processing_step = "llm_generation"
#         log_step("🤖", "Generating final answer with Mistral-7B...")
#         final_answer = generate_answer(merged_path, query)
#         st.session_state.final_answer = final_answer
#         log_step("✅", "Answer generation completed")
        
#         # STEP 8: Save and Add to Chat
#         with open(os.path.join(st.session_state.session_dir, "final_answer.txt"), "w", encoding="utf-8") as f:
#             f.write(final_answer)
        
#         st.session_state.messages.append({
#             "role": "user",
#             "content": query
#         })
        
#         st.session_state.messages.append({
#             "role": "assistant",
#             "content": final_answer,
#             "session_dir": st.session_state.session_dir
#         })
        
#         log_step("🎉", "Session completed successfully!")
#         st.session_state.processing_step = "completed"
        
#     except Exception as e:
#         error_msg = f"❌ ERROR: {str(e)}"
#         log_step("❌", error_msg)
#         st.session_state.error_message = str(e)
#         st.session_state.processing_step = "error"
#     finally:
#         st.session_state.processing = False

# # ====================== UI LAYOUT ======================
# st.set_page_config(
#     page_title="Aerospace Intelligence System",
#     page_icon="✈️",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS
# st.markdown("""
# <style>
# .chat-message {
#     padding: 1.5rem; 
#     border-radius: 0.5rem;
#     margin-bottom: 1rem;
#     display: flex;
#     flex-direction: column;
# }
# .chat-message.user {
#     background-color: #2b3540;
# }
# .chat-message.bot {
#     background-color: #1a2430;
# }
# .final-answer {
#     background-color: #1a2430;
#     border-left: 4px solid #1f77b4;
#     padding: 1.5rem;
#     margin: 1.5rem 0;
#     border-radius: 0.3rem;
# }
# .log-container {
#     background-color: #0e1117;
#     border: 1px solid #4a4a4a;
#     border-radius: 0.5rem;
#     padding: 1rem;
#     margin: 1rem 0;
#     font-family: monospace;
#     font-size: 0.9rem;
#     max-height: 400px;
#     overflow-y: auto;
#     white-space: pre-wrap;
# }
# .session-info {
#     background-color: #1a2430;
#     padding: 0.8rem;
#     border-radius: 0.3rem;
#     margin-bottom: 1rem;
#     font-size: 0.9rem;
# }
# .step-container {
#     display: flex;
#     align-items: center;
#     margin: 0.5rem 0;
# }
# .step-icon {
#     font-size: 1.2rem;
#     margin-right: 0.5rem;
#     min-width: 2rem;
# }
# .step-text {
#     flex-grow: 1;
# }
# .step-pending {
#     color: #a0a0a0;
# }
# .step-active {
#     color: #1f77b4;
#     font-weight: bold;
# }
# .step-complete {
#     color: #2ca02c;
# }
# .step-error {
#     color: #d62728;
# }
# </style>
# """, unsafe_allow_html=True)

# # Sidebar
# with st.sidebar:
#     st.title("🚀 Control Panel")
    
#     # Session controls
#     st.subheader("Session Management")
#     if st.button("🆕 New Session", use_container_width=True):
#         st.session_state.session_dir, st.session_state.session_num = initialize_session()
#         st.session_state.messages = []
#         st.session_state.logs = []
#         reset_processing_state()
#         st.success(f"✅ Created session {st.session_state.session_num}")
    
#     if st.button("🗑️ Clear Chat", use_container_width=True):
#         st.session_state.messages = []
#         st.session_state.logs = []
#         reset_processing_state()
#         st.success("✅ Chat cleared")
    
#     # Session info
#     st.subheader("Session Info")
#     if st.session_state.session_dir:
#         st.markdown(f'<div class="session-info">Session: {st.session_state.session_num}</div>', unsafe_allow_html=True)
#         st.markdown(f'<div class="session-info">Directory: {st.session_state.session_dir}</div>', unsafe_allow_html=True)
#     else:
#         st.markdown('<div class="session-info">No active session</div>', unsafe_allow_html=True)
    
#     # System info
#     st.subheader("System Info")
#     st.markdown(f'<div class="session-info">Python: {sys.version.split()[0]}</div>', unsafe_allow_html=True)
#     st.markdown(f'<div class="session-info">Status: {"🚀 Running" if not st.session_state.processing else "🟡 Processing"}</div>', unsafe_allow_html=True)

# # Main content
# st.title("✈️ Aerospace Intelligence System")

# # Display chat history
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         if message["role"] == "user":
#             st.markdown(f"**You:** {message['content']}")
#         else:
#             st.markdown("### 🤖 Aerospace Intelligence System")
#             st.markdown('<div class="final-answer">', unsafe_allow_html=True)
#             st.markdown(message["content"])
#             st.markdown('</div>', unsafe_allow_html=True)
            
#             if "session_dir" in message:
#                 session_path = os.path.relpath(message["session_dir"], start=os.getcwd())
#                 st.markdown(f"📁 *Session data: `{session_path}`*")

# # Processing status and logs
# if st.session_state.processing or st.session_state.logs:
#     st.subheader("📋 Processing Status")
    
#     # Step-by-step progress
#     steps = [
#         ("starting", "🚀 Starting"),
#         ("query_processing", "🧠 Query Processing"),
#         ("web_search", "🔍 Web Search"),
#         ("youtube_search", "📺 YouTube Search"),
#         ("web_extraction", "🌐 Web Extraction"),
#         ("youtube_extraction", "🔊 YouTube Extraction"),
#         ("merging", "🔀 Content Merging"),
#         ("llm_generation", "🤖 Answer Generation"),
#         ("completed", "✅ Completed")
#     ]
    
#     for step_id, step_name in steps:
#         step_class = "step-pending"
#         if st.session_state.processing_step == step_id:
#             step_class = "step-active"
#         elif (steps.index((step_id, step_name)) < steps.index((st.session_state.processing_step, "")) or 
#               st.session_state.processing_step == "completed"):
#             step_class = "step-complete"
        
#         if st.session_state.processing_step == "error" and step_id == st.session_state.processing_step:
#             step_class = "step-error"
        
#         st.markdown(f"""
#         <div class="step-container">
#             <span class="step-icon {'step-error' if step_class == 'step-error' else step_class}">•</span>
#             <span class="step-text {step_class}">{step_name}</span>
#         </div>
#         """, unsafe_allow_html=True)
    
#     # Display logs
#     if st.session_state.logs:
#         log_content = "\n".join(st.session_state.logs)
#         st.markdown(f'<div class="log-container">{log_content}</div>', unsafe_allow_html=True)
    
#     # Stop button during processing
#     if st.session_state.processing:
#         if st.button("⏹️ Stop Processing", key="stop_button"):
#             st.warning("🛑 Processing interrupted by user")
#             reset_processing_state()
#             st.experimental_rerun()

# # Chat input
# if not st.session_state.processing:
#     user_query = st.chat_input("Enter your aerospace query (e.g., 'why do airplanes have two engines')", key="user_input")
    
#     if user_query and user_query.strip():
#         # Start processing
#         process_query_sequentially(user_query.strip())
#         st.experimental_rerun()

# # Welcome message for empty state
# if not st.session_state.messages and not st.session_state.processing:
#     st.markdown("""
#     <div style="text-align: center; padding: 2rem; background-color: #1a2430; border-radius: 0.5rem; margin: 1rem 0;">
#         <h3>🚀 Welcome to the Aerospace Intelligence System</h3>
#         <p style="font-size: 1.1rem; color: #a0a0a0;">
#             This advanced information retrieval system combines multi-source data gathering with LLM-powered knowledge synthesis to answer complex aerospace engineering questions.
#         </p>
#         <p style="font-size: 1rem; color: #a0a0a0;">
#             To get started, click the "New Session" button in the sidebar or simply type your query below.
#         </p>
#     </div>
#     """, unsafe_allow_html=True)
    
#     st.markdown("""
#     ### 💡 Example Queries:
#     - Why do airplanes have two engines?
#     - How do jet engines work?
#     - What is ETOPS certification?
#     - How do wings generate lift?
#     - What are the advantages of composite materials in aircraft?
#     """)

# # Footer
# st.markdown("""
# <hr style="border: 1px solid #4a4a4a; margin: 2rem 0;">
# <div style="text-align: center; color: #a0a0a0; font-size: 0.9rem;">
#     Aerospace Intelligence System &copy; 2024 | Research-Enhanced Information Retrieval
# </div>
# """, unsafe_allow_html=True)