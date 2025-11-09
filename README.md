# Jharkhand Policies RAG Chatbot 📚🤖

A Retrieval-Augmented Generation (RAG) chatbot for querying Jharkhand government policy documents using Cohere AI and ChromaDB.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-latest-red.svg)
![LangChain](https://img.shields.io/badge/langchain-latest-green.svg)
![Cohere](https://img.shields.io/badge/cohere-API-orange.svg)

---

## 🎯 Overview

This RAG chatbot answers questions about Jharkhand government policies by:

1. Processing PDF policy documents
2. Creating semantic embeddings using Cohere
3. Storing embeddings in ChromaDB vector database
4. Retrieving relevant context and generating accurate answers

---

## ✨ Features

- 🔍 **Semantic Search**: Vector-based retrieval for context-aware answers
- 📄 **PDF Processing**: Automatic extraction and chunking of policy documents
- 💬 **Interactive Chat**: Clean Streamlit interface with chat history
- 📥 **Export Chat**: Download conversation history as CSV
- 🚀 **Batch Processing**: Efficient handling with rate limiting
- 💾 **Persistent Storage**: Vector database persists between sessions

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up API key in .env file
echo CO_API_KEY=your_cohere_api_key_here > .env

# 3. Add PDFs to Documents folder
# Place your policy PDFs in Documents/

# 4. Process documents
python preprocess_pdfs.py

# 5. Run the app
streamlit run app.py
```

---

## 📦 Prerequisites

- Python 3.8 or higher
- Cohere API Key ([Get it here](https://cohere.ai/))
- Internet connection for API calls

---

## 🔧 Installation

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd JHARKHAND-RAG/Chatbot
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies include:**

- `langchain` - RAG framework
- `langchain-cohere` - Cohere integration
- `langchain-chroma` - Vector database
- `streamlit` - Web interface
- `pypdf` - PDF processing
- `python-dotenv` - Environment management

### Step 3: Configure Environment

Create a `.env` file:

```bash
CO_API_KEY=your_cohere_api_key_here
```

---

## 💻 Usage

### 1. Prepare Documents

Place PDF files in the `Documents/` folder:

```
Chatbot/
└── Documents/
    ├── solar-policy.pdf
    ├── water-policy.pdf
    └── agriculture-policy.pdf
```

### 2. Process PDFs

Run the preprocessing script:

```bash
python preprocess_pdfs.py
```

**What it does:**

- Loads all PDFs from `Documents/` folder
- Splits documents into 1000-character chunks with 200-character overlap
- Generates embeddings using Cohere embed-v4.0
- Stores embeddings in ChromaDB (`chroma_persist/` folder)
- Processes in batches of 100 to respect API limits

**Expected output:**

```
Loading PDF documents...
Loaded 15 documents.
Splitting documents into chunks...
Created 245 chunks for embedding.
Initializing Cohere embeddings...
Creating or loading Chroma vector store...

Starting embedding process in batches...
Embedding batch 1/3 (100 chunks)...
Batch 1 saved successfully.
...
All batches processed successfully!
```

### 3. Run the Chatbot

Start the Streamlit app:

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`

### 4. Use the Interface

- **Ask Questions**: Type questions in the chat input
- **View History**: See past conversations
- **Reset Chat**: Clear conversation using sidebar button
- **Download Chat**: Export conversation as CSV

---

## 📁 Project Structure

```
Chatbot/
├── app.py                    # Main Streamlit application
├── preprocess_pdfs.py        # PDF processing & embedding script
├── requirements.txt          # Python dependencies
├── .env                      # API keys (create this)
├── .gitignore               # Git ignore rules
├── style.txt                # Custom CSS styling
├── logo.png                 # Application logo
├── README.md                # This file
├── Documents/               # Input PDF files
│   └── *.pdf
└── chroma_persist/          # Vector database storage
    └── chroma.sqlite3
```

---

## ⚙️ Configuration

### Processing Parameters

Edit `preprocess_pdfs.py` to customize:

```python
DOCUMENTS_DIR = "Documents"           # PDF input folder
PERSIST_DIR = "chroma_persist"        # Vector DB location
COLLECTION_NAME = "jharkhand_policies"
CHUNK_SIZE = 1000                     # Characters per chunk
CHUNK_OVERLAP = 200                   # Overlap between chunks
BATCH_SIZE = 100                      # Chunks per API batch
SLEEP_BETWEEN_BATCHES = 5             # Seconds between batches
```

### Retrieval Parameters

Edit `app.py` to customize:

```python
# In get_retrieval_chain function
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}  # Number of chunks to retrieve
)

# LLM temperature
llm = ChatCohere(
    model="command-a-03-2025",
    temperature=0.1  # Lower = more focused, Higher = more creative
)
```

---

## 🐛 Troubleshooting

### Issue: API Key Not Found

```
ERROR: CO_API_KEY not found in .env file.
```

**Solution:**

- Create `.env` file in Chatbot folder
- Add: `CO_API_KEY=your_actual_api_key`
- Ensure no quotes around the key

### Issue: No PDFs Found

```
No PDF files found in 'Documents' directory.
```

**Solution:**

- Create `Documents/` folder if missing
- Add PDF files with `.pdf` extension (lowercase)
- Verify files are readable

### Issue: Vector Database Not Found

```
Vector database not found. Please run 'preprocess.py' first
```

**Solution:**

- Run `python preprocess_pdfs.py` before starting app
- Check `chroma_persist/` folder was created
- Verify preprocessing completed successfully

### Issue: Rate Limit Errors

**Solution:**

- Increase `SLEEP_BETWEEN_BATCHES` (e.g., to 10 seconds)
- Reduce `BATCH_SIZE` (e.g., to 50)
- Check Cohere API rate limits

### Issue: Memory Errors

**Solution:**

- Reduce `BATCH_SIZE` in `preprocess_pdfs.py`
- Process fewer documents at once
- Close other applications

---

## 🛠️ Tech Stack

| Component     | Technology                 |
| ------------- | -------------------------- |
| Language      | Python 3.8+                |
| UI Framework  | Streamlit                  |
| RAG Framework | LangChain                  |
| LLM Provider  | Cohere (command-a-03-2025) |
| Embeddings    | Cohere (embed-v4.0)        |
| Vector DB     | ChromaDB                   |
| PDF Parser    | PyPDF                      |

---

## 📈 Performance Tips

### For Faster Processing

- Increase `BATCH_SIZE` if API limits allow
- Reduce `SLEEP_BETWEEN_BATCHES`

### For Better Answers

- Reduce `CHUNK_SIZE` for more granular search
- Increase retrieval `k` value for more context
- Adjust LLM `temperature` based on use case

### For Cost Optimization

- Cache embeddings (avoid reprocessing same documents)
- Use smaller `k` value in retriever
- Monitor API usage

---

## 🔄 Adding New Documents

To add new policy documents:

1. Add PDF files to `Documents/` folder
2. Run `python preprocess_pdfs.py`
3. Restart the app with `streamlit run app.py`

**Note:** Currently, the script reprocesses all documents. For production, implement incremental processing to only embed new files.

---

## 📊 How It Works

```
User Question
     ↓
Embed Query (Cohere)
     ↓
Search Vector DB (ChromaDB)
     ↓
Retrieve Top 5 Chunks
     ↓
Build Context Prompt
     ↓
Generate Answer (Cohere LLM)
     ↓
Display Response
```

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- Incremental document processing
- Hybrid search (semantic + keyword)
- Multi-session chat history
- Document metadata filtering
- Answer quality metrics

---

## 📝 License

This project is licensed under the MIT License.

---

## 👥 Author

**AbhinavSinghVishen**

- GitHub: [@AbhinavSinghVishen](https://github.com/AbhinavSinghVishen)

---

## 🙏 Acknowledgments

- [Cohere](https://cohere.ai/) for LLM and embeddings
- [LangChain](https://langchain.com/) for RAG framework
- [Streamlit](https://streamlit.io/) for UI framework
- [ChromaDB](https://www.trychroma.com/) for vector database

---

**Made with ❤️ for accessible government policy information**
