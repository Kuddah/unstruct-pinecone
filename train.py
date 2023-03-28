from io import StringIO
import sys
from typing import Dict, Optional

#from langchain.agents import load_tools
#from langchain.agents import initialize_agent
#from langchain.agents.tools import Tool
from langchain.llms import OpenAI
from langchain.document_loaders import DirectoryLoader, UnstructuredFileLoader, UnstructuredPDFLoader, OnlinePDFLoader, UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
import pinecone
from os import environ

pinecone.init(api_key = environ['PINECONE_API_KEY'], environment = environ['PINECONE_API_ENV'])

llm = OpenAI(temperature = 0.0)
# loader = DirectoryLoader('./data/', glob = '*.csv')
loader = UnstructuredFileLoader("./data/quran-complete.json")
# loader = DirectoryLoader('./data/raw_data/', glob = '*.txt')
data = loader.load()
print(f"{len(data)} CSV documents loaded")
text_splitter = CharacterTextSplitter(chunk_size = 1500, chunk_overlap = 100)
texts = text_splitter.split_documents(data)
print(f"{len(texts)} texts created")
embeddings = OpenAIEmbeddings(openai_api_key = environ['OPENAI_API_KEY'])
print("Embeddings created for each text")
index_name = "kuddahpdf"
docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name = index_name)
print("Pinecone index created and embeddings added")