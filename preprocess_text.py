import os
import time
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from utils import initialize_vector_store;

load_dotenv()
cohere_api_key = os.getenv("CO_API_KEY")

if not cohere_api_key:
    print("ERROR: CO_API_KEY not found in .env file.")
    exit(1)

TEXT_DOCUMENT = "Data\\Text_Document"
PERSIST_DIR = "chroma_persist"
COLLECTION_NAME = "jharkhand_policies"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 100
SLEEP_BETWEEN_BATCHES = 5

# Initialize vector store
vector_store = initialize_vector_store(PERSIST_DIR, cohere_api_key, COLLECTION_NAME)

print("Checking already processed files in database...")
results = vector_store.get(include=["metadatas"])
existing_docs = set()
if results and 'metadatas' in results and results['metadatas']:
    for metadata in results['metadatas']:
        if metadata and 'source' in metadata:
            existing_docs.add(metadata['source'])

print(f"Found {len(existing_docs)} already processed documents.")

# Get all text files in directory
current_docs = set()
for filename in os.listdir(TEXT_DOCUMENT):
    if filename.endswith(".txt"):
        current_docs.add(os.path.join(TEXT_DOCUMENT, filename))

print(f"Found {len(current_docs)} text files in '{TEXT_DOCUMENT}' directory.")

# Find unprocessed documents
docs_to_process = current_docs - existing_docs

if not docs_to_process:
    print("\nNo new documents to process. Your vector store is up-to-date!")
    exit(0)

print(f"\nFound {len(docs_to_process)} new document(s) to process:")
for doc in docs_to_process:
    print(f"  - {os.path.basename(doc)}")

print("\nLoading unprocessed text documents...")
documents = []
for doc_path in docs_to_process:
    print(f"\nLoading '{os.path.basename(doc_path)}'...")
    try:
        with open(doc_path, 'r', encoding='utf-8') as f:
            text_data = f.read()
        documents.append(Document(page_content=text_data, metadata={"source": doc_path}))
    except FileNotFoundError:
        print(f'File not Found: {doc_path}')
        continue
    except Exception as e:
        print(f"Error loading {doc_path}: {e}")
        continue

if not documents:
    print("No new text files could be loaded.")
    exit(1)

print(f"Loaded {len(documents)} documents.")

print("Splitting documents into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks for embedding.")

print("Initializing Cohere embeddings...")
embeddings = CohereEmbeddings(model="embed-v4.0", cohere_api_key=cohere_api_key)

print("Creating or loading Chroma vector store...")
vector_store = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME
)

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
