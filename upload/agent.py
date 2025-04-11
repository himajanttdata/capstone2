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
from database import log_query,log_user_data,check_db_for_query, get_user_history,get_user_id
import uuid



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
 
    
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    return chunks

def embed_text(directory):
    loader =TextLoader(

    file_path=directory)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_retriever(chunks, directory):
     vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=directory)
     vectorstore.persist()
     return vectorstore.as_retriever()

# summary= embed_text("summaries.txt")
# main = embed_json("cleaned_data_with_summaries.json")

# create_retriever(summary, "Summary_Large")
# create_retriever(main, "Main_Large")

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
    response: str  
    count: int     #LLM Response
    user_id: str


def grade_documents(query,documents,state):
   
    """ Determines whether the retrieved documents are relevant to the question
    by using an LLM Grader.
    If any document are not relevant to question or documents are empty - Web Search needs to be done
    If all documents are relevant to question - Web Search is not needed
    Helps filtering out irrelevant documents
    Args:
    state (dict): The current graph state
    Returns: 
    state (dict): Updates documents key with only filtered relevant documents """
   

    count = state["count"]
    if count >=1:
        return True
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
    model="Mistral-Small",#"gpt-4o-mini"
    temperature=0.1,
    max_tokens=2048,
    top_p=0.8)
    #top_k=45

    response= response.choices[0].message.content
    end = time.time()
    print("Grader took", end-start_time,"seconds")
    if "yes" in response.lower():
        result = True
    else:
        result = False
    return result

def rewrite_query(state):
    print("Entering PO")

    start_time = time.time()
    query = state["query"]
    # API_URL = "https://api-inference.huggingface.co/models/google/gemma-2-2b-it"
    # headers = {"Authorization": f"Bearer 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"}

    client= OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key="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",
    )
 
 
    response = client.chat.completions.create(
 
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI bot who optimizes and rewrites the prompt with precise, concise, and accurate responses for LLMs.\n"
                "Correct any grammatical errors or spelling mistakes without changing the meaning, intent, sentiment of the prompt."
                "If query has third-person pronouns or if the words ‘them,’ ‘they,’ ‘it,’ or any similar terms appear in a query, they refer to Comcast or Xfinity."
                "Never add your own context. Do not change the sentence type."
                "If it sounds like the complaint, do not modify anything."
                "DO NOT CHANGE ANYTHING ELSE, just rephrase it without changing the intent of it."
                
            ),
        },
        {"role": "user", "content": f"user_input: {query}. Optimize it. Give optimized prompt as the only response in next line inside double quotes and do not add any other characters."},
    ],

    model="Mistral-Small",#"gpt-4o-mini"
    temperature=0.1,
    max_tokens=2048,
    top_p=0.8
    #top_k=45
    )
    # formatted_input = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages])
 
    # payload = {
    #     "inputs": formatted_input,
    #     "parameters": {
    #         "temperature": 0.1,
    #         "top_p": 0.8,
    #         "top_k": 45
    #     }
    # }

    end_time=time.time()
    execution= end_time - start_time
    print(execution,"seconds")
    response= response.choices[0].message.content
    print("exiting PO ")

    print(response)
    
    return {"prompt":response}
 
    # response = requests.post(API_URL, headers=headers, json=payload) 
    # if response.status_code != 200:
    #     return f"Error: {response.status_code}, {response.text}" 
    # result = response.json()
    # if not result or "generated_text" not in result[0]:
    #     return "Error: Invalid response from the API"
    # generated_text = result[0]["generated_text"]
    # print(generated_text)
    # json_start = generated_text.find('"')
    # json_end = generated_text.rfind('"')
    # if json_start != -1 and json_end != -1:
    #     prompt=generated_text[json_start:json_end + 1]
    # print("Printing prompt",prompt)
    # return {"prompt":prompt}


def retrieve_from_chroma(query,directory):
    vectorstore = Chroma(persist_directory= directory, embedding_function=embeddings)
    results = vectorstore.similarity_search(query, k=5)
    page_contents = [doc.page_content for doc in results] if results else ["No relevant information found."]
    page_contents_str = "\n".join(page_contents)
    return page_contents_str

def retrieve(query):
    start_time = time.time()
    print("Entering retrieve")
    # query= state["query"]
    # summary_result = retrieve_from_chroma(query,"Summary_db")
    detailed_result =retrieve_from_chroma(query,"Detailed_db")
    print("Exiting retrieve")
    end_time = time.time()
    print("Retriever took",start_time - end_time, "seconds")
    return detailed_result

def generator(state):
    user_id = state.get("user_id", "default")
    
    print("entering generator")
    # user_id = state["user_id"]
    query=state["prompt"]
    context= state["documents"]
    start_time = time.time()

    user_history = get_user_history(user_id, limit=5)

    user_history.append({"role": "user", "content": query})

    client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key="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",
    )
 
    response = client.chat.completions.create(
    messages = [
        {
            "role": "system",
            "content": "You are a AI assistant that remenbers past interactions and provides precise, concise, and accurate responses. "
            "Use the provided context to answer the user's query accurately."
            "If the context does not contain sufficient information, state that I do not have information for the query asked, please refer the official links"
            "If any out of context questions are asked, DO NOT answer them, and tell them you can only answer xfinity and comcast related queries."
            "Answer like a friend yet professional"
            "Do not add 'in provided context' like sentences in the response."
            "Give concised answer until and unless detailed response is asked."
        },
        {
            "role": "user",
                "content": f"Context: {context}\n\nQuery: {query}"
             f"Provide an answer to the {query} using only the information from the {context}. "
             "If relevant details are missing, mention that additional information is required."
        },
    ]+user_history,
    model="gpt-4o-mini",#"gpt-4o-mini"
    temperature=0.1,
    max_tokens=2048,
    top_p=0.8
    #top_k=45
    )
    end_time=time.time()
    execution= end_time - start_time
    print("Generator took",execution,"seconds")
    response= response.choices[0].message.content
    return {"response":response}

def check_context(query):
    start =  time.time()
    flag = None
    print("Checking context")
    print("Checking in VDB")
    vectorstore = Chroma(persist_directory="Summary_db",embedding_function= embeddings)
    print("Came out of VDB")
    relevance_threshold = 1.3
    # print("Len of user query", len(user_query))
    #results = knowledge_base.search(user_query)
    context_retrieved = vectorstore.similarity_search_with_score(query, k=2)
    # print("Context retrieved")
    # print(context_retrieved)
    if context_retrieved and len(context_retrieved) > 0:
        top_doc, score = context_retrieved[0]
        # print("Retrived context score:",score, "\n", top_doc)
 
        if score > relevance_threshold:
            print("Score is high (less relevant)")
            flag= False
            print("Not in context")
        else:
            print("In context")
            flag = True
    else:
        print("Stopped here")
    end = time.time()
    print("Context Checker took", end-start, "seconds")
      
    return flag

def Agent1(state):
    if state["prompt"]==None:
        return {"category": None, "sentiment": None}
        
    print("Entering Agent 1")
    prompt = state["prompt"]
    print("Prompt in agent 1", prompt)
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
    model="gpt-4o-mini",
    temperature=0.1,
    max_tokens=2048,
    top_p=0.8
    #top_k=45
    ) 

    end_time=time.time()
    execution= end_time - start_time
    print("Agent 1 took",execution,"seconds")
    # print("Sentiment and category", response)
    result=response.choices[0].message.content 
    Classification_Result = json.loads(result) 
    # print(Classification_Result)
    category, sentiment = Classification_Result["category"], Classification_Result["sentiment"]
    
    # print("Classified category:",category)
    print("exiting agent 1")
    return {"category": category, "sentiment": sentiment}
    
    
def agent2(state):
    prompt = state['prompt']
    retrieved_docs = retrieve(prompt)
    grade = grade_documents(prompt,retrieved_docs,state)
    return{"grade": grade, "documents": retrieved_docs}
    

graph_builder = StateGraph(State)

def complaint(state):
    return {"response": "I am sorry to hear your bad experience with us 😢. Please call our toll free number 1-800-391-3000", "cached": False}

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
            "temperature": 0.1,
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
        count = state["count"]+1
        final= time.time() - start_time
        print("Rewriter took",final,"seconds")
        return {"prompt":prompt, "count": count}
    
DB_FILE2 = "querydb1.db"

def check_db_for_query(query):
    start =  time.time()
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
    end = time.time()
    print("Check db took", end-start,"seconds")
    return False

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
    lambda state: "Agent 2" if state["category"].lower()== "query" else "Complaint"
)

graph_builder.add_conditional_edges("Agent 2", lambda state: "Generator" if state["grade"] == True else "Rewrite")
graph_builder.add_edge("Rewrite","Agent 2")

graph_builder.add_edge("Complaint", END)
graph_builder.add_edge("Generator", END)
graph = graph_builder.compile()

app = Flask(__name__) #template_folder="templates",static_folder="static"

CORS(app,resources={r"/*":{"origins":"*"}})
@app.route('/')
def home():
    return render_template('newsignup.html')

@app.route("/index")
def index():
    return render_template("index.html")
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_id = data.get("user_id",None)
    # if user_id is None:
    #     return {"response": "Session error: No user_id found!"}
    # print(f"Debug: session id {user_id}")
    user_input = data.get("message", "")
    start = time.time()
    if not user_input:
        return jsonify({"error": "Message is required"}), 400
    context = check_context(user_input)
    if not context:
        return jsonify({"response": "I'm sorry I can only answer queries related to Xfinity and Comcast😊"})
    # Check for cached response
    cached_response = check_db_for_query(user_input)
    if cached_response:
        return jsonify({"response": cached_response, "category": "Cached", "sentiment":"Cached"})
    state = {
        "query": user_input,
        "context": True,
        "category": "",
        "sentiment":"",
        "response": "",
        "cached": False,
        "prompt": "",
        "documents":"",
        "grade": None,
        "count": 0,
        "user_id": ""
        
    }
    events = list(graph.stream(state))
    
    final_response = ""

    if not events:
        return jsonify({"response": "No events generated", "category": "", "sentiment":""})
    
    final_category = events[1]["Agent 1"]["category"]

    final_sentiment = events[1]["Agent 1"]["sentiment"]
    
    for event in events:
        
        data= dict(event.items())
        
        for key,value in data.items():
            

            if key== "Generator":                
                final_response = value['response']                
            elif key == "Complaint":
                final_response = value["response"]                

    log_query(user_id, user_input, final_response, final_category, final_sentiment)
    print("Logged")
    end = time.time()
    print("Total", end-start)
    return jsonify({"response": final_response, "category": final_category, "sentiment":final_sentiment})
if __name__ == "__main__":
    app.run(debug=True)

