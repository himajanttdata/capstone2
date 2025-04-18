from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.document_loaders import TextLoader
import os

from dotenv import load_dotenv
load_dotenv()

def embed_and_store_content(file_path):
    embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.environ["HF_TOKEN"],  # Replace with your actual API key
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
    loader= TextLoader("cleaned_1.txt")
    documents=loader.load()
 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
 
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="chroma")
    vectorstore.persist()
 
    return "Data embedded and stored successfully"
embed_and_store_content("chroma")