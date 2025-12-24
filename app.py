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
    if st.button("Fetch Latest IPOs"):
        # Auto-Scraping SEBI
        url = "https://www.sebi.gov.in/sebiweb/ajax/home/getnewslistinfo.jsp"
        res = requests.post(url, data={'sid': '3', 'ssid': '-1', 'smid': '0', 'nextValue': '1'})
        soup = BeautifulSoup(res.content, 'html.parser')
        st.session_state.ipo_links = [{"title": a.text.strip(), "url": a['href']} for a in soup.find_all('a', href=True) if "drhp" in a.text.lower()][:5]

    if "ipo_links" in st.session_state:
        selected_ipo = st.selectbox("Select IPO to Analyze", options=[i['title'] for i in st.session_state.ipo_links])
        if st.button("Ingest & Analyze"):
            target = next(i for i in st.session_state.ipo_links if i['title'] == selected_ipo)
            st.session_state.retriever = "loading" # Trigger ingestion
            st.session_state.current_ipo = target

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
