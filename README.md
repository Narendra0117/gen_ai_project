
# **AI Suite – RAG + Agentic Search Assistant (Groq LLM + Streamlit)**

An end-to-end **Retrieval-Augmented Generation (RAG)** and **Agentic AI Assistant** built using **Groq LLM**, **LangChain**, and **Streamlit**, deployed as an interactive web application.
The system supports contextual PDF question-answering, conversational memory, web search, summarization, and scientific research queries.

---

## 🚀 **Features**

### ✅ **1. RAG-Based Conversational PDF Q&A**

* Upload a PDF and ask any question.
* Uses **RecursiveCharacterTextSplitter**, **HuggingFace Embeddings**, and **Chroma DB**.
* Supports **history-aware retrieval** using `RunnableWithMessageHistory`.
* Provides context-rich, precise answers from the uploaded document.

---

### ✅ **2. Agentic AI with Multiple Smart Tools**

Integrated tools include:

| Tool                       | Purpose                                                    |
| -------------------------- | ---------------------------------------------------------- |
| **RAG Tool**               | Answers questions using PDF context + conversation history |
| **Summarizer Tool**        | Summarizes YouTube URLs & web URLs using LangChain loaders |
| **Wikipedia Tool**         | Retrieves relevant Wiki data                               |
| **Arxiv Tool**             | Retrieves academic papers                                  |
| **DuckDuckGo Search Tool** | Web search for real-time information                       |
| **Math Tool**              | Solves mathematical expressions using LLM Math Chain       |

---

### ✅ **3. LLM-Powered Chatbot (Groq Cloud)**

* Uses **LLaMA-3.1 8B Instant** via Groq API.
* Ultra-low latency inference.
* Integrated chat memory for long conversations.

---

### ✅ **4. Streamlit Frontend**

* Clean and interactive UI.
* Sidebar for API key input & PDF upload.
* Real-time agent responses displayed with callback streaming.

---

## 🧰 **Tech Stack**

### **Frontend**

* Streamlit

### **Backend / AI**

* Groq LLM (LLaMA-3.1-8B)
* LangChain (retrievers, chains, tools, agents)
* ChromaDB (vector storage)
* HuggingFace Embeddings (`all-MiniLM-L6-v2`)

### **Document Processing**

* PyPDFLoader
* RecursiveCharacterTextSplitter
* UnstructuredURLLoader
* YoutubeLoader

### **Deployment**

* Streamlit (local/web)
* Designed to be deployable on **Hugging Face Spaces**

---

## 🧠 **How the System Works**

### **1. Upload & Process PDF**

* PDF → chunks → embeddings → vector DB
* Retriever fetches relevant chunks
* LLM generates answer using the context

### **2. History-Aware RAG**

* Each session preserves chat history
* Improves contextual answers across conversation

### **3. Agent Tools**

* A Zero-Shot ReAct Agent decides which tool to invoke
* Example:

  * Research question → Arxiv/Wiki tool
  * URL → Summarizer tool
  * PDF query → RAG tool
  * Math expression → Math tool
  * General question → LLM

---

## 🖥️ **How to Run Locally**

1. Clone repository

```
git clone <repo-url>
cd <project-folder>
```

2. Install dependencies

```
pip install -r requirements.txt
```

3. Run Streamlit

```
streamlit run app.py
```

4. Enter your **Groq API Key** in the sidebar
5. Upload PDF → start asking questions
6. Use the text box to interact with agent tools

---

## 📌 **Project Highlights (Perfect for LinkedIn)**

* Built a **multi-agent RAG system** combining document QA, research tools, and summarization.
* Integrated **Groq LLM** for high-speed responses.
* Designed a unified Streamlit interface for easy use.
* Implemented conversational memory, tool selection, and dynamic retrieval pipelines.
