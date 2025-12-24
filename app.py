import streamlit as st
import os, requests, nest_asyncio
from bs4 import BeautifulSoup
from typing import List, Literal
from typing_extensions import TypedDict
from llama_parse import LlamaParse
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END, START

# 1. INITIAL SETUP
# nest_asyncio.apply()
st.set_page_config(page_title="IPO Analyst Agent", layout="wide")

# 2. SEBI SCRAPER (Landing Page Aware)
def get_sebi_drhp_list():
    """Scrapes SEBI listing page and resolves actual PDF links from landing pages."""
    # Correct URL for the Public Issues listing
    url = "https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=3&ssid=15&smid=10"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    # "Console logging" for the app UI
    log_placeholder = st.empty()
    
    try:
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.content, 'html.parser')
        links = []
        
        # Log basic scraping start to console/terminal
        print(f"Scraping started at: {url}")
        
        # Target table rows from SEBI's 'Public Issues' table
        rows = soup.find_all('tr')
        for row in rows:
            # SEBI uses 'points' class for the document titles
            a_tag = row.find('a', class_='points')
            if a_tag and "drhp" in a_tag.text.lower():
                title = a_tag.text.strip()
                landing_page = a_tag['href']
                
                # Resolving the landing page to get the actual PDF
                lp_res = requests.get(landing_page, headers=headers)
                lp_soup = BeautifulSoup(lp_res.content, 'html.parser')
                
                # Find the direct .pdf link on the filing page
                pdf_tag = lp_soup.find('a', href=lambda x: x and x.endswith('.pdf'))
                
                if pdf_tag:
                    pdf_url = pdf_tag['href']
                    if not pdf_url.startswith('http'):
                        pdf_url = f"https://www.sebi.gov.in{pdf_url}"
                    
                    # Console.log equivalent for terminal
                    print(f"FOUND: {title} -> {pdf_url}")
                    
                    links.append({"title": title, "url": pdf_url})
            
            if len(links) >= 5: break
            
        # UI "Console log" results
        if links:
            log_placeholder.success(f"Scrapped {len(links)} IPOs successfully. Check your terminal for details.")
        else:
            log_placeholder.warning("No IPOs found. Ensure SEBI has recent DRHP filings.")
            
        return links
    except Exception as e:
        print(f"CRITICAL SCRAPING ERROR: {e}")
        st.error(f"Scraping Error: {e}")
        return []

# Sidebar for API Keys & Auto-Fetch
with st.sidebar:
    st.header("⚙️ Configuration")
    openrouter_key = st.text_input("OpenRouter API Key", type="password")
    llama_key = st.text_input("LlamaCloud API Key", type="password")
    
    if openrouter_key and llama_key:
        os.environ["OPENROUTER_API_KEY"] = openrouter_key
        os.environ["LLAMA_CLOUD_API_KEY"] = llama_key
        st.success("API Keys Ready!")

    st.divider()
    st.header("📂 Auto-Fetch SEBI DRHPs")
    
    if st.button("Fetch Latest SEBI Filings"):
        st.session_state.ipo_links = get_sebi_drhp_list()
    
    if "ipo_links" in st.session_state and st.session_state.ipo_links:
        selected = st.selectbox("Select IPO", options=[i['title'] for i in st.session_state.ipo_links])
        if st.button("Ingest IPO Data"):
            target = next(i for i in st.session_state.ipo_links if i['title'] == selected)
            with st.spinner("LlamaParse is extracting financial tables..."):
                parser = LlamaParse(result_type="markdown", user_prompt="Extract all financial tables precisely.")
                docs = parser.load_data(target['url'])
                splits = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200).create_documents([d.text for d in docs])
                vectorstore = Chroma.from_documents(
                    documents=splits, 
                    embedding=OpenAIEmbeddings(model="openai/text-embedding-3-small", base_url="https://openrouter.ai/api/v1")
                )
                st.session_state.retriever = vectorstore.as_retriever()
                st.success(f"Ingested {selected}")

# 2. AGENTIC ENGINE (Self-Correction Loop)
class AgentState(TypedDict):
    question: str
    documents: List[str]
    relevance: str
    generation: str

def retrieve_node(state):
    return {"documents": st.session_state.retriever_obj.invoke(state["question"])}

def grade_node(state):
    # LLM Grades the retrieved data to prevent hallucinations
    llm = ChatOpenAI(model="google/gemini-2.0-flash-001", base_url="https://openrouter.ai/api/v1", openai_api_key=os.environ["OPENROUTER_API_KEY"])
    doc_txt = "\n".join([d.page_content for d in state["documents"]])
    check = llm.invoke(f"Does this context answer: '{state['question']}'? Reply ONLY 'yes' or 'no'. Context: {doc_txt}")
    return {"relevance": "yes" if "yes" in check.content.lower() else "no"}

def generate_node(state):
    llm = ChatOpenAI(model="google/gemini-2.0-flash-001", base_url="https://openrouter.ai/api/v1", openai_api_key=os.environ["OPENROUTER_API_KEY"])
    context = "\n\n".join([d.page_content for d in state["documents"]])
    res = llm.invoke(f"You are a Senior Financial Analyst. Answer using context: {context}\n\nQuestion: {state['question']}")
    return {"generation": res.content}

# Build LangGraph
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade", grade_node)
workflow.add_node("generate", generate_node)
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges("grade", lambda x: "generate" if x["relevance"] == "yes" else END)
workflow.add_edge("generate", END)
agent_app = workflow.compile()

# 3. CHAT INTERFACE
st.title("👨‍💻 Financial RAG Agent")
if "messages" not in st.session_state: st.session_state.messages = []

# Handle Ingestion
if "retriever" in st.session_state and st.session_state.retriever == "loading":
    with st.spinner(f"Parsing {st.session_state.current_ipo['title']}..."):
        parser = LlamaParse(result_type="markdown", user_prompt="Extract financial tables accurately.")
        docs = parser.load_data(st.session_state.current_ipo['url'])
        splits = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200).create_documents([d.text for d in docs])
        vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(base_url="https://openrouter.ai/api/v1", openai_api_key=os.environ["OPENROUTER_API_KEY"]))
        st.session_state.retriever_obj = vectorstore.as_retriever()
        st.session_state.retriever = "ready"
        st.success("Ready for Analysis!")

# Chat History UI
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("Ask about the IPO financial health..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    if "retriever_obj" in st.session_state:
        with st.chat_message("assistant"):
            result = agent_app.invoke({"question": prompt})
            ans = result.get("generation", "I couldn't find enough relevant data in the DRHP to answer this accurately.")
            st.markdown(ans)
            st.session_state.messages.append({"role": "assistant", "content": ans})
    else:
        st.error("Please select and ingest an IPO from the sidebar first.")
