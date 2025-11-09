import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma #to avoid deprication warning
from langchain_cohere import CohereEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import initialize_vector_store

# --- Load API key ---
load_dotenv()
cohere_api_key = os.getenv("CO_API_KEY")

if not cohere_api_key:
    print("ERROR: CO_API_KEY not found in .env file.")
    exit(1)

# --- Settings ---
DOCUMENTS_DIR = "Documents"
PERSIST_DIR = "chroma_persist"
COLLECTION_NAME = "jharkhand_policies"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 100   # how many chunks to embed at once
SLEEP_BETWEEN_BATCHES = 5  # seconds pause between batches

# Accessing vector database
vector_store = initialize_vector_store(PERSIST_DIR, cohere_api_key, COLLECTION_NAME)

# Fix the metadata access code
print("Checking already processed files in database..")
# The correct way to access metadata from Chroma's get() method
results = vector_store.get(include=["metadatas"])
existing_docs = set()
if results and 'metadatas' in results and results['metadatas']:
    for metadata in results['metadatas']:
        if metadata and 'source' in metadata:
            existing_docs.add(metadata['source'])

print(f"Found {len(existing_docs)} already processed documents.")

# Get list of currently present docs 
current_docs = set()
for filename in os.listdir(DOCUMENTS_DIR):
    if filename.endswith(".pdf"):
        # We create the full path just like the loader does, for a perfect match.
        current_docs.add(os.path.join(DOCUMENTS_DIR, filename))

print(f"Found {len(current_docs)} PDF files in '{DOCUMENTS_DIR}' directory.")

# Check whether new unprocessed files present to pre-process
docs_to_process = current_docs - existing_docs

if not docs_to_process:
    print("\nNo new documents to process. Your vector store is up-to-date!")
    exit(0)

print(f"\nFound {len(docs_to_process)} new document(s) to process:")



# --- Step 1: Load only unprocessed PDFs ---
print("Loading unprocessed PDF documents...")
documents = []
for doc_path in docs_to_process:
    print(f"\nLoading '{os.path.basename(doc_path)}'...")
    loader = PyPDFLoader(doc_path)
    # .load() returns a list of pages, so we extend our main list.
    documents.extend(loader.load())

if not documents:
    print(f"No new PDF files could be loaded.")
    exit(1)

print(f"Loaded {len(documents)} documents.")

# --- Step 2: Split into Chunks ---
print("Splitting documents into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks for embedding.")

# --- Step 3: Initialize Cohere embeddings ---
print("Initializing Cohere embeddings...")
embeddings = CohereEmbeddings(model="embed-v4.0", cohere_api_key=cohere_api_key)

# --- Step 4: Create or load Chroma store ---
print("Creating or loading Chroma vector store...")
vector_store = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME
)

# --- Step 5: Batch processing with pause ---
print("\nStarting embedding process in batches...")
total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

for batch_num in range(total_batches):
    start = batch_num * BATCH_SIZE
    end = min((batch_num + 1) * BATCH_SIZE, len(chunks))
    batch = chunks[start:end]

    print(f"Embedding batch {batch_num + 1}/{total_batches} ({len(batch)} chunks)...")
    try:
        vector_store.add_documents(batch)
        print(f"Batch {batch_num + 1} saved successfully.")
    except Exception as e:
        print(f"Error in batch {batch_num + 1}: {e}")
        print("Retrying after delay...")
        time.sleep(10)
        continue

    if batch_num < total_batches - 1:
        print(f"Sleeping {SLEEP_BETWEEN_BATCHES} seconds before next batch...")
        time.sleep(SLEEP_BETWEEN_BATCHES)

print("\nAll batches processed successfully!")
print(f"Vector store saved in '{PERSIST_DIR}' and ready to use in your app.")
