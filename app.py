import os
from icecream import ic
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings,HuggingFaceInstructEmbeddings
from templates import css,bot_template,user_template
from templates import css,bot_template,user_template
from langchain.chat_models import ChatOpenAI
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.vectorstores import FAISS

from langchain.memory import ConversationBufferMemory


def getDocText(pdf_file):
    text=''
    for pdf in pdf_file:
        pdf_read=PdfReader(pdf)
        for pdf_page in pdf_read.pages:
            text+= pdf_page .extract_text()
    return text

def getTextChunks(text):
    textsplit = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = textsplit.split_text(text)
    return chunks

def getstoredinDB(textchunks):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    ic(len(textchunks))
    ic(embeddings)
    vector_store = FAISS.from_texts(textchunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store  

def load_llm():
     llm = CTransformers(
        model = "llama-2-7b-chat.Q4_0.gguf", 
        model_file="llama-2-7b-chat.Q4_0.gguf", 
        max_new_tokens = 1024,
        temperature = 0.5,
        config = {'context_length' : 2048},
        )
     return llm

def getconversation(dbStore): 
    llm=load_llm()
    memory = ConversationBufferMemory(
    memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=dbStore.as_retriever(),
        memory=memory
    )
   
    
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    st.set_page_config(page_title='UniqueBot',page_icon=':book:')

    st.write(css,unsafe_allow_html=True)

    st.header("Unique Bot")
    user_question=st.text_input("Ask your questions")
    if user_question:
        handle_userinput(user_question)      
    with st.sidebar:
        st.subheader('My Documents')
        pdf_file=st.file_uploader('Upload PDFs',accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner("Processing......"):
                if pdf_file:
                    text=getDocText(pdf_file)
                    
                    textchunks=getTextChunks(text)
                    
                    dbStore=getstoredinDB(textchunks)
                    st.session_state.conversation = getconversation(
                    dbStore)
    
if __name__==  '__main__':
    main()
