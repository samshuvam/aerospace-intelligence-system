# Aerospace Intelligence System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Research](https://img.shields.io/badge/Research-Domain%20Specific%20IR-orange)](research/)

A research-enhanced information retrieval system that answers complex aerospace engineering questions by intelligently gathering and synthesizing information from web pages and YouTube videos using advanced algorithms and large language models.

## Research Innovations

This project implements several research-backed innovations:

- **Domain-Specific TrustRank Algorithm**: Adapts the seminal TrustRank algorithm (Gyöngyi et al., 2004) for aerospace source credibility assessment
- **Knowledge Graph Enhanced Merging**: Implements cross-document entity coreference (Erera et al., 2023) for multi-source information integration
- **Context-Aware Query Expansion**: Uses BERT embeddings with domain adaptation for query understanding
- **Multi-Modal Information Retrieval**: Fuses textual and video content for comprehensive knowledge synthesis

## Features

- **Natural Language Query Processing**: Understands complex aerospace engineering questions
- **Multi-Source Information Gathering**: Searches web and YouTube for relevant content
- **Source Credibility Assessment**: Ranks sources based on authority and relevance using TrustRank
- **Content Deduplication & Integration**: Merges information from multiple sources using knowledge graphs
- **Technical Answer Generation**: Produces comprehensive, structured answers using Mistral-7B
- **Session Management**: Maintains conversation history and context across interactions
- **Research-Grade Algorithms**: Implements state-of-the-art information retrieval algorithms

## Installation
  Prerequisites
  Python 3.10 or newer
  Windows 10/11 (or Linux/macOS)
  16GB+ RAM (for Mistral-7B model)
  NVIDIA GPU recommended (but not required)
  FFmpeg installed system-wide


##Setup Steps

# Clone the repository
git clone https://github.com/YOUR_USERNAME/aerospace-intelligence-system.git
cd aerospace-intelligence-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Download Whisper model (first run)
python -c "import whisper; whisper.load_model('base')"

# Place Mistral model in model/ directory
# Download from: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF
# File: mistral-7b-instruct-v0.2.Q4_K_M.gguf

