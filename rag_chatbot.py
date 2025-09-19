import streamlit as st
import google.generativeai as genai
import os
import pandas as pd

# ============ 1. SIDEBAR CONTROLS ============
st.sidebar.header("⚙ Settings")

api_key = st.sidebar.text_input("🔑 Gemini API Key", type="password")
model_name = st.sidebar.selectbox("📦 Select Model", ["gemini-1.5-flash", "gemini-1.5-pro"])
temperature = st.sidebar.slider("🌡 Temperature", 0.0, 1.0, 0.7, 0.1)
max_tokens = st.sidebar.slider("📝 Max Tokens", 100, 2048, 512, 50)

if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.messages = []

if st.sidebar.button("🆕 New Chat"):
    st.session_state.messages = []

# ============ 2. INITIALIZE ============
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("📚 Knowledge-Powered Rag Chatbot (Gemini)")

# ✅ File uploader hamesha visible rahega
uploaded_file = st.file_uploader("📂 Upload Knowledge Base", type=["csv", "txt"])

# Default KB
knowledge_chunks = ["Knowledge base not found."]

# Agar user ne file upload ki
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            text_data = []
            for col in df.columns:
                text_data.extend(df[col].dropna().astype(str).tolist())
            knowledge_chunks = [chunk.strip() for chunk in text_data if chunk.strip()]
            st.success(f"✅ Loaded {len(knowledge_chunks)} entries from uploaded CSV")

        elif uploaded_file.name.endswith(".txt"):
            content = uploaded_file.read().decode("utf-8")
            knowledge_chunks = [line.strip() for line in content.splitlines() if line.strip()]
            st.success(f"✅ Loaded {len(knowledge_chunks)} lines from uploaded TXT")

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")

# Agar API key missing hai to warning
if not api_key:
    st.warning("Please enter your Gemini API key in the sidebar to start.")
    st.stop()

# ============ 3. GEMINI CONFIG ============
genai.configure(api_key=api_key)
llm = genai.GenerativeModel(model_name)

# ============ 4. SEMANTIC SEARCH ============
def semantic_search(query, top_k=3):
    query_words = query.lower().split()
    scored_chunks = []
    for chunk in knowledge_chunks:
        score = sum(chunk.lower().count(word) for word in query_words)
        scored_chunks.append((score, chunk))
    scored_chunks.sort(reverse=True, key=lambda x: x[0])
    top_chunks = [chunk for score, chunk in scored_chunks if score > 0]
    return top_chunks[:top_k] if top_chunks else ["No relevant information found in knowledge base."]

# ============ 5. RAG PIPELINE ============
def rag_pipeline(question):
    context_docs = semantic_search(question)
    context = "\n".join(context_docs)
    prompt = f"""
You are a helpful AI assistant. Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer in a clear and human-like way:
"""
    response = llm.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens
        )
    )
    return response.text

# ============ 6. MAIN CHAT UI ============
user_input = st.chat_input("✍️ Ask your question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.spinner("Thinking..."):
        answer = rag_pipeline(user_input)
    st.session_state.messages.append({"role": "assistant", "content": answer})

# Display history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"👨‍💻 **You:** {msg['content']}")
    else:
        st.markdown(f"🔎 **AI:** {msg['content']}")
