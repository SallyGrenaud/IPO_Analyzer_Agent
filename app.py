import streamlit as st
import os, requests, asyncio
from bs4 import BeautifulSoup
from typing import List
from typing_extensions import TypedDict
from llama_parse import LlamaParse
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_groq import ChatGroq # UPDATED: Changed from OpenAI to Groq
from langchain_huggingface import HuggingFaceEmbeddings # UPDATED: Free local embeddings
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END, START

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="IPO Analyst Agent", layout="wide", page_icon="📈")
st.title("📈 Indian IPO Analyst Pro")

# 2. FIXED SEBI SCRAPER (Landing Page & Iframe Aware)
def get_sebi_drhp_list():
    url = "https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=3&ssid=15&smid=10"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.content, 'html.parser')
        links = []
        rows = soup.find_all('tr', {'role': 'row'})
        for row in rows:
            a_tag = row.find('a', class_='points')
            if a_tag and "drhp" in a_tag.text.lower():
                title = a_tag.text.strip()
                landing_page = a_tag['href']
                lp_res = requests.get(landing_page, headers=headers)
                lp_soup = BeautifulSoup(lp_res.content, 'html.parser')
                iframe = lp_soup.find('iframe')
                pdf_url = None
                if iframe and 'src' in iframe.attrs:
                    src = iframe['src']
                    pdf_url = src.split('file=')[-1] if 'file=' in src else src
                if not pdf_url:
                    pdf_tag = lp_soup.find('a', href=lambda x: x and x.endswith('.pdf'))
                    pdf_url = pdf_tag['href'] if pdf_tag else None
                if pdf_url:
                    if not pdf_url.startswith('http'):
                        pdf_url = f"https://www.sebi.gov.in{pdf_url}"
                    links.append({"title": title, "url": pdf_url})
            if len(links) >= 5: break
        return links
    except Exception as e:
        st.error(f"Scraping Error: {e}")
        return []

# 3. AGENTIC ENGINE (Self-Correction Loop)
class AgentState(TypedDict):
    question: str
    documents: List[str]
    relevance: str
    generation: str

# Helper to initialize Groq LLM
def get_llm():
    return ChatGroq(
        model_name="llama-3.3-70b-versatile", # Groq's most powerful model
        groq_api_key=os.environ["GROQ_API_KEY"],
        temperature=0
    )

def retrieve_node(state):
    docs = st.session_state.retriever_obj.invoke(state["question"])
    return {"documents": docs}

def grade_node(state):
    llm = get_llm()
    context = "\n".join([d.page_content for d in state["documents"]])
    prompt = f"Does this context answer: '{state['question']}'? Reply ONLY 'yes' or 'no'. Context: {context}"
    score = llm.invoke(prompt).content.lower()
    return {"relevance": "yes" if "yes" in score else "no"}

def generate_node(state):
    llm = get_llm()
    context = "\n\n".join([d.page_content for d in state["documents"]])
    prompt = f"You are a Senior Financial Analyst. Answer accurately using context: {context}\n\nQuestion: {state['question']}"
    return {"generation": llm.invoke(prompt).content}

workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade", grade_node)
workflow.add_node("generate", generate_node)
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges("grade", lambda x: "generate" if x["relevance"] == "yes" else END)
workflow.add_edge("generate", END)
agent_app = workflow.compile()

# 4. SIDEBAR CONFIGURATION
with st.sidebar:
    st.header("⚙️ Configuration")
    groq_key = st.text_input("Groq API Key", type="password")
    llama_key = st.text_input("LlamaCloud API Key", type="password")
    
    if groq_key and llama_key:
        os.environ["GROQ_API_KEY"] = groq_key
        os.environ["LLAMA_CLOUD_API_KEY"] = llama_key
        st.success("API Keys Ready!")

    st.divider()
    st.header("📂 Data Source")
    if st.button("Fetch Latest SEBI Filings"):
        st.session_state.ipo_links = get_sebi_drhp_list()
    
    if "ipo_links" in st.session_state and st.session_state.ipo_links:
        selected = st.selectbox("Select IPO", options=[i['title'] for i in st.session_state.ipo_links])
        if st.button("Ingest IPO Data"):
            if not os.environ.get("GROQ_API_KEY") or not os.environ.get("LLAMA_CLOUD_API_KEY"):
                st.error("❌ API Keys are missing!")
            else:
                target = next(i for i in st.session_state.ipo_links if i['title'] == selected)
                with st.spinner(f"LlamaParse is extracting tables from {selected}..."):
                    try:
                        parser = LlamaParse(result_type="markdown")
                        docs = parser.load_data(target['url'])
                        if not docs:
                            st.error("LlamaParse returned no data.")
                            st.stop()
        
                        splits = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150).create_documents([d.text for d in docs])

                        # UPDATED: Using HuggingFace Embeddings (free, fast, and local)
                        with st.spinner("Initializing HuggingFace Embeddings..."):
                            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                        
                        vectorstore = Chroma.from_documents(
                            documents=splits, 
                            embedding=embeddings
                        )
                        
                        st.session_state.retriever_obj = vectorstore.as_retriever()
                        st.success(f"✅ Ingested {selected}")
                        
                    except Exception as e:
                        st.error(f"Error: {e}")

# 5. CHAT INTERFACE
if "messages" not in st.session_state: st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("Ask about debt, revenue, or risk factors..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    if "retriever_obj" in st.session_state:
        with st.chat_message("assistant"):
            with st.spinner("Groq is reasoning..."):
                result = agent_app.invoke({"question": prompt})
                ans = result.get("generation", "Agent determined the retrieved data was irrelevant.")
                st.markdown(ans)
                st.session_state.messages.append({"role": "assistant", "content": ans})
    else:
        st.error("Please select and ingest an IPO from the sidebar first.")
