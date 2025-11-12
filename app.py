import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime
from utils import initialize_vector_store

from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains import ConversationalRetrievalChain
from langchain_cohere import ChatCohere

load_dotenv()


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

    vector_store = initialize_vector_store(persist_directory, _cohere_api_key)

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
    style_path = "style.txt"
    if os.path.exists(style_path):
        with open(style_path, "r", encoding="utf-8") as f:
            style_content = f.read()
        st.markdown(f"{style_content}", unsafe_allow_html=True)
    else:
        st.warning("style.txt not found. Using default style.")


def main():
    st.set_page_config(page_title="Jharkhand Policies ChatBot", page_icon="📚")
    inject_external_style()
    Jharkhand_state_logo = "logo.png"

    col1, col2 = st.columns([1, 8])

    with col1:
        st.image(Jharkhand_state_logo, width=64)

    with col2:
        st.markdown(
            """
            <h1 id="jharkhand-policies-chat-bot">Jharkhand Policies ChatBot</h1>
            """,
            unsafe_allow_html=True
        )




    cohere_api_key = os.getenv("CO_API_KEY")
    if not cohere_api_key:
        st.error("CO_API_KEY not found")
        st.stop()

    if "history" not in st.session_state:
        st.session_state.history = []

    with st.sidebar:
        st.title("Menu")
        if st.button("🔄 Reset Chat"):
            st.session_state.history = []
            st.rerun()

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

    retrieval_chain = get_retrieval_chain(cohere_api_key)

    if retrieval_chain is None:
        st.stop()

     
    for message in st.session_state.history:
        with st.chat_message(message["Role"]):
            st.markdown(message["Message"])
    
    if user_question := st.chat_input("Ask a question about any policy..."):
        st.session_state.history.append(
            {"Role": "user", "Message": user_question, "Timestamp": datetime.now()}
        )
        with st.chat_message("user"):
            st.markdown(user_question)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chat_history = [(msg["Role"], msg["Message"]) for msg in st.session_state.history]
                response = retrieval_chain.invoke({
                    "question": user_question,
                    "chat_history": chat_history
                })
                response_output = response["answer"]
            
                st.session_state.history.append(
                    {
                        "Role": "assistant",
                        "Message": response_output,
                        "Timestamp": datetime.now(),
                    }
                )
                st.markdown(response_output)
                st.rerun()


if __name__ == "__main__":
    main()