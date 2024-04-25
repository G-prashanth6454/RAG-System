import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import google.generativeai as genai
from langchain_text_splitters import NLTKTextSplitter

st.header('RAG System on “Leave No Context Behind” Paper')

loader = PyPDFLoader("pdf.pdf")

pages = loader.load_and_split()

page = "".join([p.page_content for p in pages])

f = open('LangChain_RAG_API_KEY.txt')
key = f.read()
genai.configure(api_key=key)

text_splitter = NLTKTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(pages)

embedding_model = GoogleGenerativeAIEmbeddings(google_api_key = key, model = "models/embedding-001")

db = Chroma.from_documents(chunks, embedding_model, persist_directory = "./chroma_db_")
db.persist()
db_connection = Chroma(persist_directory = "./chroma_db_", embedding_function = embedding_model)

retriever = db_connection.as_retriever(search_kwargs={"k":5})

model = genai.GenerativeModel('gemini-1.5-pro-latest')

chat = model.start_chat(history=[])

user_input_1 = st.text_input('Enter Your Question here : ')

user_input = page + user_input_1

response = chat.send_message(user_input)

if st.button('Answer'):
    st.subheader('User Query : ')
    st.write(user_input_1)
    st.subheader('Systems Response : ')
    st.write(response.text)