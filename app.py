import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import os
import pinecone
from docx import Document

import base64
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


def get_doc_text(doc_docs):
    text = ""
    for doc in doc_docs:
        doc_reader = Document(doc)
        for para in doc_reader.paragraphs:
            text += para.text
    return text

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = Pinecone.from_texts(texts=text_chunks, embedding=embeddings, index_name='chatwithpdfs1')
    return vectorstore

def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k')
    llm = ChatOpenAI(model_name='gpt-4')

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i in range(len(st.session_state.chat_history)-2, -2, -2):
        st.write(user_template.replace(
            "{{MSG}}", st.session_state.chat_history[i].content), unsafe_allow_html=True) 
        st.write(bot_template.replace(
            "{{MSG}}", st.session_state.chat_history[i+1].content), unsafe_allow_html=True) 

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple documents",
                       page_icon=":books:")

    logo_base64 = get_image_base64("logo.jpg") 

    st.markdown(
        f"""
        <div style="position: absolute; top: 0px; left: 0px;">
            <img src="data:image/jpg;base64,{logo_base64}" width="210">
            <div style="color: #aaa; font-size: 1.3em; text-align: right;">Beta</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Logout hyperlink
    if st.session_state.logged_in:
        st.markdown("""
            <a style='display: block; text-align: right;' href='?logout' target='_self'>Logout</a>
            """, unsafe_allow_html=True)


    st.header("Chat with multiple documents :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        uploaded_files = st.file_uploader(
            "Upload your PDFs or DOCX files here and click on 'Process'", 
            type=["pdf", "docx"], 
            accept_multiple_files=True
        )

        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf and doc text
                raw_text = ""
                for uploaded_file in uploaded_files:
                    if uploaded_file.type == "application/pdf":
                        raw_text += get_pdf_text([uploaded_file])
                    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                        raw_text += get_doc_text([uploaded_file])

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # setup pinecone
                PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
                PINECONE_ENV = os.getenv('PINECONE_ENV')
                pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
        
def login():
    load_dotenv()
    st.title("Login to the App")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        correct_username = os.getenv("LOGIN_USERNAME")
        correct_password = os.getenv("LOGIN_PASSWORD")

        # Check if the username and password are correct
        if username == correct_username and password == correct_password:
            st.session_state.logged_in = True
            st.success("Logged in successfully")
            st.experimental_rerun()
        else:
            st.error("Incorrect username or password")


if __name__ == "__main__":
    if "logged_in" in st.session_state:
        if st.session_state.logged_in:
            main()
        else:
            login()
    else:
        login()
