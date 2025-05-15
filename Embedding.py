import os
import torch
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings

model_name = "BAAI/bge-large-en-v1.5"
encode_kwargs = {'normalize_embeddings': True} 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_norm = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={'device': DEVICE},
    encode_kwargs=encode_kwargs)

loader_raw = DirectoryLoader('Raw_data', glob="*.txt")
loader_pdf = DirectoryLoader('pdf_file_data', glob="*.txt")
docs_raw = loader_raw.load()
docs_pdf = loader_pdf.load()

docs = docs_raw + docs_pdf
print(len(docs))


text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
texts = text_splitter.split_documents(docs)
print(len(texts))

embedding = model_norm
persist_directory = 'db'
vectordb = Chroma.from_documents(documents=texts,embedding=embedding,persist_directory=persist_directory)

#Kiểm tra embedding
retriever = vectordb.as_retriever(search_kwargs={"k": 100})
docs = retriever.get_relevant_documents("Which universities are located in Pittsburgh?")
print(docs)