from io import StringIO
import sys
from typing import Dict, Optional
from os import environ

from langchain import OpenAI, VectorDBQA
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents.tools import Tool
#from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Pinecone
import pinecone

pinecone.init(
api_key = environ['PINECONE_API_KEY'],  # find at app.pinecone.io
environment = environ['PINECONE_API_ENV']   # next to api key in console
)

index = pinecone.Index("kuddahpdf")

llm = OpenAI(temperature = 0.0)
embeddings = OpenAIEmbeddings(openai_api_key = environ['OPENAI_API_KEY'])
chain = load_qa_chain(llm, chain_type = "stuff")
docsearch = Pinecone(index, embeddings.embed_query, "text")

while True:
  query = input("Query: ")
  docs = docsearch.similarity_search(query, include_metadata = True)
  print(chain.run(input_documents = docs, question = query))
