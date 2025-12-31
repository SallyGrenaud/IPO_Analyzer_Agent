# 📈 IPO Analyzer: Agentic RAG for SEBI Filings

A high-performance, **Agentic RAG (Retrieval-Augmented Generation)** system designed to automate the analysis of Indian IPO Draft Red Herring Prospectuses (DRHPs). This tool scrapes live data from SEBI, processes 300+ page financial documents using hybrid parsing, and provides a self-correcting AI agent interface for investment research.

## 🚀 Key Features

* **Automated SEBI Scraper**: Real-time crawling of the SEBI Public Issues portal to fetch the latest DRHPs.
* **Hybrid Ingestion Engine**: Dual-layer extraction using **LlamaParse** (for high-accuracy table extraction) with an automated fallback to **PyMuPDF** for local processing.
* **Agentic Self-Correction**: Built with **LangGraph**, the system employs a "Grade-then-Generate" loop. An evaluator agent checks the relevance of retrieved financial data before the analyst agent formulates a response, significantly reducing hallucinations.
* **Persistent Disk Caching**: Optimized ChromaDB implementation with a local storage layer. Once an IPO is analyzed, it is cached to disk, enabling sub-second load times for future queries without redundant API costs.
* **Ultra-Low Latency Inference**: Powered by **Groq** (Llama 3.3-70B), achieving token generation speeds exceeding 500 tokens/sec.

## 🛠️ Tech Stack

* **LLM Orchestration**: LangGraph, LangChain
* **Inference Engine**: Groq (Llama 3.3-70B-Versatile)
* **Data Ingestion**: LlamaParse (LlamaIndex), PyMuPDF
* **Vector Database**: ChromaDB (with Disk Persistence)
* **Embeddings**: HuggingFace (sentence-transformers/all-MiniLM-L6-v2)
* **Frontend**: Streamlit
* **Web Scraping**: BeautifulSoup4, Requests

## 📋 Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/[YOUR_GITHUB_USERNAME]/ipo-analyst-pro.git
cd ipo-analyst-pro

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Run the Application:**
```bash
streamlit run app.py

```


4. **Environment Configuration:**
Enter your `GROQ_API_KEY` and `LLAMA_CLOUD_API_KEY` in the Streamlit sidebar or set them as environment variables.

## 🧠 System Architecture

The application follows a modular "Agentic" flow:

1. **Retrieve**: Pulls relevant chunks from the persistent ChromaDB index based on user query.
2. **Grade**: A specialized LLM node evaluates if the chunks contain the specific financial data required.
3. **Re-route/Generate**: If relevant, the Analyst node synthesizes a response. If irrelevant, the system prompts for more specific data or informs the user of the data gap.
