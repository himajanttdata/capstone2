from flask import Flask, request, jsonify, render_template
from typing import List
from flask_cors import CORS
from typing_extensions import TypedDict, Annotated
from langgraph.graph.state import StateGraph, START, END
import time
import json
import sqlite3
from openai import OpenAI
import os
import requests
from dotenv import load_dotenv
load_dotenv()
import re
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.document_loaders import TextLoader, JSONLoader


from langgraph.prebuilt import ToolNode
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.tools.retriever import create_retriever_tool

import database



embeddings = OpenAIEmbeddings(openai_api_base="https://models.inference.ai.azure.com", api_key="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", model='text-embedding-3-small')


def preprocess_dataset(docs_list):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=700,
        chunk_overlap=50,
        disallowed_special=()
    )
    doc_splits = text_splitter.split_documents(docs_list)
    return doc_splits


def embed_json(directory):
    loader = JSONLoader(

    file_path=directory, jq_schema=".", text_content=False)
 
    # content_key="terms",
    # metadata_func=lambda record: {"links": record.get("links", [])} 
# 2) Load into Documents
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    return chunks

    # vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=directory)
    # vectorstore.persist()
    # return vectorstore.as_retriever()

def embed_text(directory):
    loader =TextLoader(

    file_path=directory)
 
    # content_key="terms",
    # metadata_func=lambda record: {"links": record.get("links", [])} 
# 2) Load into Documents
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    return chunks

    # vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=directory)
    # vectorstore.persist()
    # return vectorstore.as_retriever()


summary = embed_text("summaries.txt")

data_and_summary = embed_json("cleaned_data_with_summaries.json")

def create_retriever(chunks, directory):
     vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=directory)
     vectorstore.persist()
     return vectorstore.as_retriever()

class State(TypedDict):
    query: str   #User input
    context: bool       #Contxet checker
    cached: bool        #SQL DB
    category: str       #Query or Complaint
    sentiment: str      #P,Neg,Neutral
    prompt: str         #Optimized prompt
    vector_cached: bool #Vector DB Cached
    documents: str
    grade: bool         #Grader
    #web_search: str
    response: str       #LLM Response


# Adjust the persist directory paths as needed.
# direct_answer_retriever = create_retriever(
#     chunks=data_and_summary_documents,
#     directory="path/to/direct_answer_db"
# )
summary_retriever = create_retriever(
    chunks=summary,
    directory="Summary_db"
)
detailed_retriever = create_retriever(
    chunks=data_and_summary,
    directory="Detailed_db"
)


# direct_answer_tool = create_retriever_tool(
    
#     direct_answer_retriever,
#     "direct answer retriever",
#     "Search and return a direct answer from the primary vector database."
# )
# summary_tool = create_retriever_tool(
#     summary_retriever,
#     "summary retriever",
#     "Search and return a summary from the secondary vector database."
# )

# detailed_tool = create_retriever_tool(
#     detailed_retriever,
#     "detailed retriever",
#     "Search and return a detailed answer from the tertiary vector database using the summary."
# )


# #add direct retriever to the list
# tools = [summary_tool, detailed_tool]

# tool_node = ToolNode(tools=tools)


# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.pydantic_v1 import BaseModel, Field
# from langchain_openai import ChatOpenAI
# # Data model for LLM output format
# class GradeDocuments(BaseModel):
#     """Binary score for relevance check on retrieved documents."""
    
#     binary_score: str = Field(description="Documents are relevant to the query, 'yes' or 'no'")


def grade_documents(query,documents):
    print("Entering grade documents")
    """ Determines whether the retrieved documents are relevant to the question
    by using an LLM Grader.
    If any document are not relevant to question or documents are empty - Web Search needs to be done
    If all documents are relevant to question - Web Search is not needed
    Helps filtering out irrelevant documents
    Args:
    state (dict): The current graph state
    Returns: 
    state (dict): Updates documents key with only filtered relevant documents """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")

    
    question = query
    documents = documents
    start_time = time.time()
    client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key="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"
    )
 
    response = client.chat.completions.create(
    messages = [
        {
            "role": "system",
            "content": """You are an expert grader assessing relevance of a retrieved document to a user question.
                    Follow these instructions for grading:
                  - If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
                  - Your grade should be either 'yes' or 'no' to indicate whether the document is relevant to the question or not."""
        },
        {
            "role": "user",
                "content": f"""Retrieved document:{documents} User question:{question}"""
            
        },
    ],
    model="Phi-4",#"gpt-4o-mini"
    temperature=0.1,
    max_tokens=2048,
    top_p=0.8)
    #top_k=45

    response= response.choices[0].message.content
    print("Result",response)
    print("Exiting grade documents")

    if "yes" in response.lower():
        result = True
    else:
        result = False
    return result

def rewrite_query(state):

    
    print("Entering rewrite query")
    query = state["query"]
    documents= state["documents"]
    start_time = time.time()
    API_URL = "https://api-inference.huggingface.co/models/google/gemma-2-2b-it"
    headers = {"Authorization": f"Bearer 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"}
 

 
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI bot who optimizes and rewrites the prompt with precise, concise, and accurate responses for LLMs.\n"
                "Use the provided input to generate the prompt properly with to give it as input to another LLM which answers on Comcast and Xfinity.\n"
                "If query has third-person pronouns or if the words ‘them,’ ‘they,’ ‘it,’ or any similar terms appear in a query, they refer to Comcast or Xfinity."
                "Design the prompt so that other models can retrieve the proper information from the document.\n"
                "Never add your own context apart from the user's query context.\n"
                "Add pointers if needed.\n"
                "Do not change the meaning, intent,sentiment of the query.\n"
                'Output MUST  BE only prompt that has been optimized '
                "Ensure rewritten prompt remains faithful to original intent"
            ),
        },
        {"role": "user", "content": f"user_input: {query}. Optimize it. Give the only optimized prompt as the response in"},
    ]
 
    formatted_input = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages])
 
    payload = {
        "inputs": formatted_input,
        "parameters": {
            "temperature": 0.3,
            "top_p": 0.8,
            "top_k": 45
        }
    }
 
    response = requests.post(API_URL, headers=headers, json=payload) 
    if response.status_code != 200:
        return f"Error: {response.status_code}, {response.text}" 
    result = response.json()
    if not result or "generated_text" not in result[0]:
        return "Error: Invalid response from the API"
    generated_text = result[0]["generated_text"]
    print(generated_text)
    pattern = r"Prompt:\s*(.*)"
    match = re.search(pattern, generated_text)
    if match:
        prompt = match.group(1)
        print(prompt)
    else:
        prompt = query
        print(prompt)
    print("Exiting rewrite query")
    return {"prompt": prompt}


def retrieve_from_chroma(query,directory):
    vectorstore = Chroma(persist_directory= directory, embedding_function=embeddings)
    results = vectorstore.similarity_search(query, k=5)
    print(results)
    page_contents = [doc.page_content for doc in results] if results else ["No relevant information found."]
    page_contents_str = "\n".join(page_contents)
    return page_contents_str

def retrieve(query):

    print("Entering retrieve")
    # query= state["query"]
    summary_result = retrieve_from_chroma(query,"Summary_db")
    detailed_result =retrieve_from_chroma(query,"Detailed_db")
    print("Exiting retrieve")
    return detailed_result

def generator(state):

    print("entering generator")
    
    query=state["prompt"]
    context= state["documents"]
    start_time = time.time()
    client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key="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",
    )
 
    response = client.chat.completions.create(
    messages = [
        {
            "role": "system",
            "content": "You are a AI assistant that provides precise, concise, and accurate responses. "
            "Use the provided context to answer the user's query accurately."
            "If the context does not contain sufficient information, state that I do not have information for the query asked, please refer the official links"
            "If any out of context questions are asked, DO NOT answer them, and tell them you can only answer xfinity and comcast related queries."
            "Answer like a friend yet professional"
        },
        {
            "role": "user",
                "content": f"Context: {context}\n\nQuery: {query}"
             f"Provide a detailed yet concise answer to the {query} using only the information from the {context}. "
             "If relevant details are missing, mention that additional information is required."
        },
    ],
    model="Phi-4",#"gpt-4o-mini"
    temperature=0.1,
    max_tokens=2048,
    top_p=0.8
    #top_k=45
    )
    end_time=time.time()
    execution= end_time - start_time
    print(execution,"seconds")
    response= response.choices[0].message.content
    print("exiting generator")

    print(response)
    
    return {"response":response}


def check_context(query):
    print("Checking context")
    vectorstore = Chroma(persist_directory="jsonKB",embedding_function= embeddings)
    relevance_threshold = 1.1
    # print("Len of user query", len(user_query))
    #results = knowledge_base.search(user_query)
    context_retrieved = vectorstore.similarity_search_with_score(query, k=3)
    if context_retrieved and len(context_retrieved) > 0:
        top_doc, score = context_retrieved[0]
        print("Retrived context score:",score, "\n", top_doc)
 
        if score > relevance_threshold:
            print("Score is high (less relevant)")
            flag= False
        else:
            flag = True
    print("exiting check context")
      
    return flag
    
   
DB_FILE = "user_data.db"
DB_FILE2 = "querydb.db"
 
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name TEXT,
            email TEXT,
            password TEXT
        )
    """)
    conn.commit()
    conn.close()
 
def init_db2():
    conn = sqlite3.connect(DB_FILE2)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS querydb (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            lm_response TEXT,
            category TEXT,
            sentiment TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()
init_db2()
 
def log_user_data(user_name: str, email: str, password: str):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO user_data (user_name, email, password)
        VALUES (?, ?, ?)
    """, (user_name, email, password))
    conn.commit()
    conn.close()
 
def log_query(query: str, lm_response: str, category: str, sentiment: str):
    conn = sqlite3.connect(DB_FILE2)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO querydb (query, lm_response, category, sentiment)
        VALUES (?, ?, ?, ?)
    """, (query, lm_response, category, sentiment))
    conn.commit()
    conn.close()
 

# Initialize databases on startup


def check_db_for_query(query):
    print("Entering check db")
    conn = sqlite3.connect(DB_FILE2)
    cursor = conn.cursor()
    cursor.execute("SELECT lm_response FROM querydb WHERE query = ?", (query,))
    row = cursor.fetchone()
    conn.close()
    result = row[0] if row else None
    
    if result:
        print("Exiting check db")
        return result
    print("Exiting check db")
    return False


def Agent1(state):
    if state["prompt"]==None:
        return {"category": None, "sentiment": None}
        
    print("Entering Agent 1")
    prompt = state["prompt"]
    print("Prompt received by agent 1",prompt)
    start_time = time.time()
    client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key="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",
    )
 
    response = client.chat.completions.create(
    messages = [
        {
            "role": "system",
            "content": """
                    You are a helpful and expert text analyzer.
                    Your goal is to analyze user queries and
                    1. Determine the overall sentiment of the user query. You are free to label them into different categories of emotions.
                    2. Classify the query as either 'Complaint' or 'Query'.

                    Output MUST be a valid JSON with the following keys:
                    "sentiment" and "category"
                    """
        },
        {
            "role": "user",
                "content": f"""
                User Query:"{prompt}"
                Please return a JSON object in the format:
                {{"sentiment":, "category":}}
                """

        },
    ],
    model="Mistral-small",
    temperature=0.3,
    max_tokens=2048,
    top_p=0.8
    #top_k=45
    ) 

    end_time=time.time()
    execution= end_time - start_time
    print(execution,"seconds")
    print("Sentiment and category", response)
    result=response.choices[0].message.content 
    Classification_Result = json.loads(result) 
    print(Classification_Result)
    category, sentiment = Classification_Result["category"], Classification_Result["sentiment"]
    
    print("Classified category:",category)
    print("exiting agent 1")
    return {"category": category, "sentiment": sentiment}
    
    
def agent2(state):
    prompt = state['prompt']
    retrieved_docs = retrieve(prompt)
    grade = grade_documents(prompt,retrieved_docs)
    return{"grade": grade, "documents": retrieved_docs}
    

graph_builder = StateGraph(State)

def complaint(state):
    print("Please contact customer support")

def rewriter(state):
    start_time = time.time()
    query = state["query"]
    API_URL = "https://api-inference.huggingface.co/models/google/gemma-2-2b-it"
    headers = {"Authorization": f"Bearer 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"}
 

 
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI bot who optimizes and rewrites the prompt with precise, concise, and accurate responses for LLMs.\n"
                "Use the provided input to generate the prompt properly to give it as input to another LLM which answers on Comcast and Xfinity.\n"
                "If query has third-person pronouns or if the words ‘them,’ ‘they,’ ‘it,’ or any similar terms appear in a query, they refer to Comcast or Xfinity."
                "Design the prompt so that other models can retrieve the proper information from the document.\n"
                "Never add your own context apart from the user's query context.\n"
                "Add pointers if needed.\n"
                "Do not change the meaning, intent,sentiment of the query.\n"
                'Output MUST be a valid JSON with the following key: "prompt".'
                "Ensure rewritten prompt remains faithful to original intent"
            ),
        },
        {"role": "user", "content": f"user_input: {query}. Optimize it. Give the response in JSON format."},
    ]
 
    formatted_input = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages])
 
    payload = {
        "inputs": formatted_input,
        "parameters": {
            "temperature": 0.3,
            "top_p": 0.8,
            "top_k": 45
        }
    }
 
    response = requests.post(API_URL, headers=headers, json=payload) 
    if response.status_code != 200:
        return f"Error: {response.status_code}, {response.text}" 
    result = response.json()
    if not result or "generated_text" not in result[0]:
        return "Error: Invalid response from the API"
    generated_text = result[0]["generated_text"]
    json_start = generated_text.find("{")
    json_end = generated_text.rfind("}")
    if json_start != -1 and json_end != -1:
        prompt= json.loads(generated_text[json_start:json_end + 1])
        prompt=prompt['prompt']
        print(prompt)
        print("In Rewriterrrrr")
        return {"prompt":prompt}




graph_builder.add_node("Prompt Optimizer",rewrite_query)
graph_builder.add_node("Agent 1",Agent1)
graph_builder.add_node("Agent 2", agent2)
graph_builder.add_node("Generator",generator)
graph_builder.add_node("Complaint",complaint)
graph_builder.add_node("Rewrite",rewriter)



graph_builder.add_edge(START, "Prompt Optimizer")
graph_builder.add_edge("Prompt Optimizer", "Agent 1")
graph_builder.add_conditional_edges(
    "Agent 1",
    lambda state: "Agent 2" if state["category"]== "query" else "Complaint"
)

graph_builder.add_conditional_edges("Agent 2", lambda state: "Generator" if state["grade"] == True else "Rewrite")
graph_builder.add_edge("Rewrite","Agent 2")

graph_builder.add_edge("Complaint", END)
graph_builder.add_edge("Generator", END)
graph = graph_builder.compile()


while True:

    user_input = input()
    start_time = time.time()
    if not user_input:
        print("Bye")

    response = check_db_for_query(user_input)
    if response:
        continue
    else:
         state = {
            "query": user_input,
            "context": True,
            "category": "",
            "sentiment":"",
            "response": "",
            "cached": False,
            "prompt": "",
            "documents":"",
            "grade": None
            
        }
         events = list(graph.stream(state))
    end_time = time.time()
    print(end_time-start_time, "seconds")

        
    
    
   
    

