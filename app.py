import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime
from utils import initialize_vector_store

# --- LangChain Imports ---
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains import ConversationalRetrievalChain
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

    # Use the shared function to initialize vector store
    vector_store = initialize_vector_store(persist_directory, _cohere_api_key)

    # Define the RAG prompt template
    prompt_template = """
    # You are an assistant that MUST answer in English ONLY using ONLY the information in the provided Context.
    # Your answer should be clear, concise, and directly address the question.
    # If the answer cannot be found exactly in the Context, reply with something like:
    # "I couldn't find any Jharkhand government policy related to your query. Please try rephrasing your question or specify a department or scheme for more accurate results."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Initialize the LLM and retrieval chain
    llm = ChatCohere(model="command-a-03-2025", api_key=_cohere_api_key, temperature=0.5)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    retrieval_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=False
    )
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
                chat_history = [(msg["Role"], msg["Message"]) for msg in st.session_state.history]
                response = retrieval_chain.invoke({
                    "question": user_question,
                    "chat_history": chat_history
                })
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