import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime

# --- LangChain Imports ---
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma #to avoid deprication warning
from langchain_cohere import ChatCohere, CohereEmbeddings

# --- Load Environment Variables ---
# Make sure you have a .env file with your CO_API_KEY
load_dotenv()


# --- Caching Mechanism for Performance ---
@st.cache_resource
def get_retrieval_chain(_cohere_api_key):
    """
    Loads a cached retrieval chain from the persisted Chroma vector database.
    If the database is not found, it returns None.
    """
    persist_directory = 'chroma_persist'

    if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
        st.warning(
            "Vector database not found. Please run 'preprocess.py' first to build the database."
        )
        return None

    # Initialize Cohere embeddings
    embeddings = CohereEmbeddings(model="embed-v4.0", cohere_api_key=_cohere_api_key)

    # Load existing Chroma vector store
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="jharkhand_policies"
    )

    # Define the RAG prompt template
    prompt_template = """
    # You are an assistant that MUST answer using ONLY the information in the provided Context.
    # Your answer should be clear, concise, and directly address the question.
    # If the answer cannot be found exactly in the Context, reply with:
    # "It seems that this specific information isn't covered in the provided government policies."

    Context:
    {context}

    Question:
    {input}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "input"])

    # Initialize the LLM and retrieval chain
    llm = ChatCohere(model="command-a-03-2025", api_key=_cohere_api_key, temperature=0.1)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

def inject_external_style():
    style_path = "style.txt"  # Path to your text file
    if os.path.exists(style_path):
        with open(style_path, "r", encoding="utf-8") as f:
            style_content = f.read()
        # Inject the text exactly as is (useful for CSS)
        st.markdown(f"{style_content}", unsafe_allow_html=True)
    else:
        st.warning("style.txt not found. Using default style.")


# --- Main Application Logic ---
def main():
    # Set up the Streamlit page configuration
    st.set_page_config(page_title="Jharkhand Policies ChatBot", page_icon="📚")
    # Apply green theme CSS
    inject_external_style()
    Jharkhand_state_logo = "logo.png"

    col1, col2 = st.columns([1, 8])  # Adjust ratio for spacing

    with col1:
        st.image(Jharkhand_state_logo, width=64)  # Logo size

    with col2:
        st.markdown(
            """
            <h1 id="jharkhand-policies-chat-bot">Jharkhand Policies ChatBot</h1>
            """,
            unsafe_allow_html=True
        )




    # Check for the Cohere API key
    cohere_api_key = os.getenv("CO_API_KEY")
    if not cohere_api_key:
        st.error("CO_API_KEY not found")
        st.stop()

    # Initialize session state for storing conversation history
    if "history" not in st.session_state:
        st.session_state.history = []

    # --- Sidebar for Controls and History Download ---
    with st.sidebar:
        st.title("Menu")
        if st.button("🔄 Reset Chat"):
            st.session_state.history = []
            st.rerun()  # Reruns the app to clear the chat display

        if st.session_state.history:
            st.markdown("---")
            st.markdown("### Download Chat History")
            # Convert history to a DataFrame for easy CSV conversion
            df = pd.DataFrame(
                st.session_state.history, columns=["Role", "Message", "Timestamp"]
            )
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

    # Load the retrieval chain (will be cached after the first run)
    retrieval_chain = get_retrieval_chain(cohere_api_key)

    # Stop execution if the chain couldn't be created (e.g., no PDFs found)
    if retrieval_chain is None:
        st.stop()

    # --- Modern Chat Interface ---
     
    # Display past messages from session state
    for message in st.session_state.history:
        with st.chat_message(message["Role"]):
            st.markdown(message["Message"])
    
    # Get user input from the chat input box at the bottom of the screen
    if user_question := st.chat_input("Ask a question about any policy..."):
        # Add user message to history and display it
        st.session_state.history.append(
            {"Role": "user", "Message": user_question, "Timestamp": datetime.now()}
        )
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # Get and display the bot's response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = retrieval_chain.invoke({"input": user_question})
                response_output = response["answer"]
            
                # Add bot response to history and display it
                st.session_state.history.append(
                    {
                        "Role": "assistant",
                        "Message": response_output,
                        "Timestamp": datetime.now(),
                    }
                )
                st.markdown(response_output)
                st.rerun()


# --- Application Entry Point ---
if __name__ == "__main__":
    main()