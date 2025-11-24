import os
import time
from llama_cpp import Llama
from tqdm import tqdm

MODEL_PATH = os.path.join("model", "mistral-7b-instruct-v0.2.Q4_K_M.gguf")

# At the top of llm.py, add this function:
def truncate_context(text, max_tokens=3500):
    """Simple truncation that works reliably"""
    words = text.split()
    truncated = []
    current_length = 0
    
    for word in words:
        current_length += len(word) + 1  # +1 for space
        if current_length > max_tokens * 4:  # Rough estimate: 4 chars per token
            break
        truncated.append(word)
    
    return " ".join(truncated) + "..." if len(truncated) < len(words) else " ".join(truncated)

def load_model():
    print("LOADING MISTRAL-7B MODEL (this may take 1-2 minutes)...")
    start = time.time()
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,
        n_threads=8,
        n_gpu_layers=0,
        verbose=False,
        use_mmap=True,
        use_mlock=False
    )
    print(f"MODEL LOADED SUCCESSFULLY in {time.time()-start:.1f} seconds")
    return llm

def generate_answer(merged_path, original_query):
    llm = load_model()
    
    with open(merged_path, "r", encoding="utf-8") as f:
        knowledge_base = f.read()
    
    print("\n GENERATING COMPREHENSIVE ANSWER...")

    # In the generate_answer function, add this before creating the prompt:
# Truncate knowledge base to fit context window
    if len(knowledge_base) > 14000:  # ~3500 tokens
        knowledge_base = truncate_context(knowledge_base, 3200)
        print("⚠️  Knowledge base truncated to fit context window")
    
    # Enhanced prompt for detailed, informative response
    prompt = f"""<s>[INST]
You are an expert aerospace engineer with decades of experience. Your task is to provide a comprehensive, detailed, and informative answer to the user's question using ONLY the information provided in the knowledge base below.

GUIDELINES:
1. Provide a thorough explanation with technical depth appropriate for an educated audience
2. Organize your answer with clear sections and logical flow
3. Include specific examples, statistics, and engineering principles where relevant
4. Explain both the "how" and "why" behind aerospace concepts
5. If the knowledge base lacks information on certain aspects, acknowledge this limitation
6. Use professional terminology but explain complex concepts clearly
7. Aim for a comprehensive answer of 500+ words with multiple supporting points

KNOWLEDGE BASE:
{knowledge_base}

USER QUESTION: {original_query}

STRUCTURED RESPONSE FORMAT:
1. Introduction: Brief overview of the topic
2. Technical Explanation: Detailed breakdown of engineering principles
3. Historical Context: Evolution of the technology or concept
4. Safety Considerations: Relevant safety aspects and protocols
5. Modern Applications: Current implementations and innovations
6. Conclusion: Summary of key points

Begin your comprehensive response below:
[/INST]"""
    
    print("⏱️ Processing with Mistral-7B (this may take 1-2 minutes for detailed response)...")
    
    output = llm(
        prompt,
        max_tokens=1500,
        stop=["</s>", "[INST]", "USER:", "ASSISTANT:"],
        temperature=0.3,
        top_p=0.92,
        repeat_penalty=1.1,
        echo=False
    )
    
    return output["choices"][0]["text"].strip()













# import os
# import time
# from llama_cpp import Llama
# from tqdm import tqdm

# MODEL_PATH = os.path.join("model", "mistral-7b-instruct-v0.2.Q4_K_M.gguf")

# def load_model():
#     print("🧠 Loading Mistral-7B model (this may take 1-2 minutes)...")
#     start = time.time()
#     llm = Llama(
#         model_path=MODEL_PATH,
#         n_ctx=4096,
#         n_threads=8,  
#         n_gpu_layers=0,
#         verbose=False
#     )
#     print(f"Model loaded in {time.time()-start:.1f} seconds")
#     return llm

# def generate_answer(merged_path, original_query):
#     llm = load_model()
    
#     with open(merged_path, "r", encoding="utf-8") as f:
#         knowledge_base = f.read()
    
#     if len(knowledge_base) > 12000:
#         knowledge_base = knowledge_base[:12000] + "... [TRUNCATED]"
    
#     prompt = f"""<s>[INST]
# You are an aerospace engineering expert. Answer the question using ONLY the provided knowledge base. 
# Be precise, factual, and cite sources when possible. If the knowledge base doesn't contain relevant information, say "I don't have sufficient information".

# KNOWLEDGE BASE:
# {knowledge_base}

# QUESTION: {original_query}
# [/INST]"""
    
#     print("⏱️ Generating response (may take 30-60 seconds)...")
#     output = llm(
#         prompt,
#         max_tokens=1024,
#         stop=["</s>", "[INST]", "USER:"],
#         temperature=0.3,
#         top_p=0.9,
#         echo=False
#     )
    
#     return output["choices"][0]["text"].strip()