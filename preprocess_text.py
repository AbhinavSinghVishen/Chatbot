import os
import time
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

load_dotenv()
cohere_api_key = os.getenv("CO_API_KEY")

if not cohere_api_key:
    print("ERROR: CO_API_KEY not found in .env file.")
    exit(1)

TEXT_DOCUMENT = "Data\\Text_Document\\data.txt"
PERSIST_DIR = "chroma_persist"
COLLECTION_NAME = "jharkhand_policies"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 100
SLEEP_BETWEEN_BATCHES = 5

print("Loading text file...")
text_data = ''
try:
    with open(TEXT_DOCUMENT, 'r', encoding='utf-8') as f:
        text_data = f.read()
except FileNotFoundError:
    print(f'File not Found: {TEXT_DOCUMENT}')
    exit(1)
except Exception as e:
    print("Some error occured!", e)
    exit(1)

documents = [Document(page_content=text_data)]

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
