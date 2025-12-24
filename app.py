import streamlit as st
import os, requests, asyncio, fitz  # fitz is PyMuPDF
from bs4 import BeautifulSoup
from typing import List
from typing_extensions import TypedDict
from llama_parse import LlamaParse
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_groq import ChatGroq 
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END, START

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="IPO Analyst Agent", layout="wide", page_icon="📈")
st.title("📈 Indian IPO Analyst Pro (Resilient Edition)")

# 2. SEBI SCRAPER
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

# 3. AGENTIC ENGINE
class AgentState(TypedDict):
    question: str
    documents: List[str]
    relevance: str
    generation: str

def get_llm():
    return ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=os.environ["GROQ_API_KEY"], temperature=0)

def retrieve_node(state):
    return {"documents": st.session_state.retriever_obj.invoke(state["question"])}

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

# 4. SIDEBAR & INGESTION
with st.sidebar:
    st.header("⚙️ Configuration")
    groq_key = st.text_input("Groq API Key", type="password")
    llama_key = st.text_input("LlamaCloud API Key (Optional)", type="password", help="If empty, local extraction will be used.")
    
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
        st.success("Groq Ready!")

    st.divider()
    if st.button("Fetch Latest SEBI Filings"):
        st.session_state.ipo_links = get_sebi_drhp_list()
    
    if "ipo_links" in st.session_state and st.session_state.ipo_links:
        selected = st.selectbox("Select IPO", options=[i['title'] for i in st.session_state.ipo_links])
        if st.button("Ingest IPO Data"):
            target = next(i for i in st.session_state.ipo_links if i['title'] == selected)
            text_content = ""
            
            # Use a status container for a professional loading experience
            with st.status(f"Processing {selected}...", expanded=True) as status:
                
                # --- STEP 1: DOWNLOAD ---
                status.write("📥 Fetching PDF from SEBI servers...")
                try:
                    response = requests.get(target['url'], timeout=30)
                    response.raise_for_status()
                    with open("temp.pdf", "wb") as f: 
                        f.write(response.content)
                except Exception as e:
                    st.error(f"Download failed: {e}")
                    st.stop()

                # --- STEP 2: EXTRACTION ---
                if llama_key:
                    try:
                        status.write("🔍 Parsing with LlamaParse (High Accuracy)...")
                        os.environ["LLAMA_CLOUD_API_KEY"] = llama_key
                        parser = LlamaParse(result_type="markdown")
                        docs = parser.load_data("temp.pdf")
                        text_content = "\n\n".join([d.text for d in docs])
                    except Exception as e:
                        status.write(f"⚠️ LlamaParse failed ({e}). Falling back to local...")

                if not text_content:
                    status.write("📄 Extracting text locally using PyMuPDF...")
                    doc = fitz.open("temp.pdf")
                    text_content = "\n".join([page.get_text() for page in doc])

                # --- STEP 3: EMBEDDING & VECTOR DB ---
                if text_content.strip():
                    status.write("⚙️ Splitting text into chunks...")
                    splits = RecursiveCharacterTextSplitter(
                        chunk_size=1500, 
                        chunk_overlap=150
                    ).create_documents([text_content])
                    
                    status.write("🧠 Generating embeddings (HuggingFace)...")
                    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    
                    status.write("💾 Finalizing Vector Database...")
                    vectorstore = Chroma.from_documents(
                        documents=splits, 
                        embedding=embeddings
                    )
                    
                    st.session_state.retriever_obj = vectorstore.as_retriever()
                    status.update(label="✅ Ingestion Complete!", state="complete", expanded=False)
                    st.success(f"Successfully Ingested {selected}")
                else:
                    status.update(label="❌ Extraction Failed", state="error")
                    st.error("Could not extract any text from the document.")

# 5. CHAT INTERFACE
if "messages" not in st.session_state: st.session_state.messages = []
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("Ask about debt, revenue, or risk factors..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    if "retriever_obj" in st.session_state:
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                result = agent_app.invoke({"question": prompt})
                ans = result.get("generation", "Context not relevant.")
                st.markdown(ans)
                st.session_state.messages.append({"role": "assistant", "content": ans})
    else:
        st.error("Please ingest an IPO first.")
