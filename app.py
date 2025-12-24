import streamlit as st
import os, requests, asyncio, fitz
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
st.set_page_config(page_title="IPO Analyst Pro", layout="wide", page_icon="📈")
st.title("📈 Indian IPO Analyst Agent")
st.caption("Automated SEBI Scraper + Persistent RAG + Groq Llama 3.3")

# 2. SEBI SCRAPER (No Limit + Landing Page Logic)
def get_sebi_drhp_list():
    url = "https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=3&ssid=15&smid=10"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.content, 'html.parser')
        links = []
        rows = soup.find_all('tr')
        for row in rows:
            a_tag = row.find('a', class_='points')
            if a_tag and "drhp" in a_tag.text.lower():
                links.append({"title": a_tag.text.strip(), "url": a_tag['href']})
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

def get_llm():
    return ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=os.environ["GROQ_API_KEY"], temperature=0)

def retrieve_node(state):
    return {"documents": st.session_state.retriever_obj.invoke(state["question"])}

def grade_node(state):
    llm = get_llm()
    doc_txt = "\n".join([d.page_content[:1000] for d in state["documents"]])
    # More lenient grader to avoid "Context not relevant"
    prompt = f"""
    Is the following context useful for answering a financial or business question about an IPO?
    Respond ONLY 'yes' or 'no'.
    Context: {doc_txt}
    Question: {state['question']}
    """
    check = llm.invoke(prompt).content.lower()
    return {"relevance": "yes" if "yes" in check else "no"}

def generate_node(state):
    llm = get_llm()
    context = "\n\n".join([d.page_content for d in state["documents"]])
    prompt = f"""
    You are a Senior IPO Financial Analyst. Use the context provided to answer the user's question.
    Context: {context}
    
    If the user asks if they should 'apply', provide a data-driven summary of Pros and Cons.
    Do not give direct financial advice.
    Question: {state['question']}
    """
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

# 4. SIDEBAR & CACHED INGESTION
with st.sidebar:
    st.header("⚙️ Configuration")
    groq_key = st.text_input("Groq API Key", type="password")
    llama_key = st.text_input("LlamaCloud API Key (Optional)", type="password")
    
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
        st.success("Groq Ready!")

    st.divider()
    if st.button("Fetch Latest SEBI Filings"):
        st.session_state.ipo_links = get_sebi_drhp_list()
    
    if "ipo_links" in st.session_state and st.session_state.ipo_links:
        selected = st.selectbox("Select IPO", options=[i['title'] for i in st.session_state.ipo_links])
        
        if st.button("Ingest & Analyze"):
            target = next(i for i in st.session_state.ipo_links if i['title'] == selected)
            cache_dir = f"./cache_{selected.replace(' ', '_')[:15]}"
            
            with st.status(f"Processing {selected}...", expanded=True) as status:
                # CHECK CACHE FIRST
                if os.path.exists(cache_dir):
                    status.write("🚀 Found in local cache. Loading...")
                    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    vectorstore = Chroma(persist_directory=cache_dir, embedding_function=embeddings)
                    st.session_state.retriever_obj = vectorstore.as_retriever()
                    status.update(label="✅ Loaded from Cache!", state="complete", expanded=False)
                else:
                    # FETCH ACTUAL PDF LINK
                    status.write("📥 Resolving PDF URL...")
                    lp_res = requests.get(target['url'], headers={'User-Agent': 'Mozilla/5.0'})
                    lp_soup = BeautifulSoup(lp_res.content, 'html.parser')
                    iframe = lp_soup.find('iframe')
                    pdf_url = iframe['src'].split('file=')[-1] if iframe and 'src' in iframe.attrs else target['url']
                    if not pdf_url.startswith('http'): pdf_url = f"https://www.sebi.gov.in{pdf_url}"
                    
                    # DOWNLOAD
                    status.write("📂 Downloading PDF...")
                    pdf_data = requests.get(pdf_url).content
                    with open("temp.pdf", "wb") as f: f.write(pdf_data)

                    # EXTRACTION
                    text_content = ""
                    if llama_key:
                        try:
                            status.write("🔍 Extracting with LlamaParse...")
                            os.environ["LLAMA_CLOUD_API_KEY"] = llama_key
                            docs = LlamaParse(result_type="markdown").load_data("temp.pdf")
                            text_content = "\n\n".join([d.text for d in docs])
                        except: status.write("⚠️ LlamaParse failed. Falling back...")

                    if not text_content:
                        status.write("📄 Local extraction with PyMuPDF...")
                        doc = fitz.open("temp.pdf")
                        text_content = "\n".join([page.get_text() for page in doc])

                    # EMBEDDING & PERSISTENCE
                    status.write("🧠 Building Vector Database...")
                    splits = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150).create_documents([text_content])
                    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=cache_dir)
                    st.session_state.retriever_obj = vectorstore.as_retriever()
                    status.update(label="✅ Analysis & Caching Complete!", state="complete", expanded=False)
                
            st.success(f"Successfully Ingested {selected}")

# 5. CHAT INTERFACE
if "messages" not in st.session_state: st.session_state.messages = []
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("Ask about financials, risk, or 'should I apply?'"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    
    if "retriever_obj" in st.session_state:
        with st.chat_message("assistant"):
            with st.spinner("Analyzing DRHP..."):
                result = agent_app.invoke({"question": prompt})
                ans = result.get("generation", "Agent determined the retrieved data was irrelevant to this specific query.")
                st.markdown(ans)
                st.session_state.messages.append({"role": "assistant", "content": ans})
    else:
        st.error("Please select and ingest an IPO from the sidebar first.")
