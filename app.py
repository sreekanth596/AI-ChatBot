import time

#from dotenv import load_dotenv
import huggingface_hub
import streamlit as st
#from pyPDF2 import PdfReader
from PyPDF2 import PdfReader
from PyPDF2 import PdfFileReader
from langchain.text_splitter import CharacterTextSplitter, TextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings,HuggingFaceInstructEmbeddings
from templates import css,bot_template,user_template
from langchain.chat_models import ChatOpenAI
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
#from langchain.vectorstores import FIASS
from sentence_transformers import SentenceTransformer, util
from datasets import Dataset


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
        chunk_overlap=200,
        length_function=len
    )
    chunks = textsplit.split_text(text)
    return chunks

def getstoredinDB(textchunks):
   ''' #emmbeddings=OpenAIEmbeddings()
    emmbeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore=FAISS.from_texts(texts=textchunks,embedding=emmbeddings)
    return vectorstore'''
   
   '''local_model_path = r"C:/Users/Muraai/Desktop/AIBot"'''
   '''local_model_path=r"C:/Users/Muraai/Desktop/AIBot/llama-2-7b-chat.Q4_0.gguf"
   embeddings = HuggingFaceEmbeddings(model_path=local_model_path)
   vectorstore = FAISS.from_texts(texts=textchunks, embedding=embeddings)'''
   '''model_name = "llama-2-7b-chat.Q4_0"
   model_kwargs = {"device": "cuda"}
   #embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
   embeddings = HuggingFaceEmbeddings(model_name=model_name)
   vectorstore = Chroma.from_documents(documents=textchunks, embedding=embeddings, persist_directory="chroma_db")'''
   model_path = "llama-2-7b-chat.Q4_0.gguf"
   model = SentenceTransformer(model_path)
   embeddings = model.encode(textchunks)
   dataset = Dataset.from_dict({"text": textchunks, "embedding": embeddings})
   return dataset
   #return vectorstore

def load_llm():
     llm = CTransformers(
        model = "llama-7b.ggmlv3.Q4_0.bin",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
        )
     return llm

def getconversation(dbStore):
    '''llm=ChatOpenAI()
    llm = huggingface_hub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    memory=ConversationBufferMemory( 
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=dbStore.as_retriever(),
        memory=memory
        )'''
    
    llm = load_llm()
    #memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True) 
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=dbStore.as_retriever())

def handle_userinput(user_question):
      #response = st.session_state.conversation({'question': user_question})
            st.session_state.chat_history =['chat_history']

      #for i, message in enumerate(st.session_state.chat_history):
        #if i % 2 == 0:
            #st.write(user_template.replace(
             #   "{{MSG}}", message.content), unsafe_allow_html=True)
        #else:
           # st.write(bot_template.replace(
               # "{{MSG}}", message.content), unsafe_allow_html=True)
            st.write(user_template.replace(
                "{{MSG}}", user_question), unsafe_allow_html=True)
        #else:
            st.write(bot_template.replace(
                "{{MSG}}", 'No data found'), unsafe_allow_html=True)



def main():
    #load_dotenv()
    st.set_page_config(page_title='ChatBot',page_icon=':book:')

    st.write(css,unsafe_allow_html=True)

    st.header("Chat with Bot")
    user_question=st.text_input("Ask your questions")
    if user_question:
        handle_userinput(user_question)
    #vectorstores.save_local("faiss_index")
    with st.sidebar:
        st.subheader('My Documents')
        pdf_file=st.file_uploader('Upload PDFs',accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner("Processing......"):
                if pdf_file:
                    text=getDocText(pdf_file)
                    st.write(text)
                    textchunks=getTextChunks(text)
                    st.write(textchunks)
                    #x=len(pdf_file)
                    #st.success(f"{len(pdf_file)} Files Uploaded Successfully.")
                    dbStore=getstoredinDB(textchunks)
                    #st.write(dbStore)
                    #convo=getconversation(dbStore)
                
            
                    

        



if __name__==  '__main__':
    main()