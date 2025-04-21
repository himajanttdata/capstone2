#!/usr/bin/env python
# coding: utf-8

# In[1]:


from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import requests
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.document_loaders import TextLoader
import time


# In[2]:


API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment"
HEADERS = {"Authorization": "Bearer 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"}


# In[3]:


def classify_sentiment(text):
    print("Entering classify_sentiment")
    response = requests.post(API_URL, headers=HEADERS, json={"inputs": text})
    result = response.json()
    
    if "LABEL_0" in result[0][0]["label"]:
        print("Exiting classify_sentiment")
        return "Complaint"
    else:
        print("Exiting classify_sentiment")
        return "Query"


# In[4]:


def embed_and_store_content(content):
    print("Entering embed and store")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(content)
 
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key="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", model_name="BAAI/bge-base-en-v1.5"
    )
 
    vectorstore = Chroma.from_documents(chunks, embeddings)
    print("Came out of embed and store")
    return vectorstore


# In[5]:


def query_retrieval_pipeline(state):
    print("Entering query_retrieval_pipeline")
    query = state["messages"][-1].content
 
    # Load the cleaned text
    loader = TextLoader("cleaned_data.txt")
    text_documents = loader.load()
 
    # Embed and store the content
    vectorstore = embed_and_store_content(text_documents)
 
    # Query the vectorstore
    results = vectorstore.similarity_search(query)
 
    if results:
        print("Came out ofquery_retrieval_pipeline")
        return {"response": results[0].page_content}
    else:
         print("Came out ofquery_retrieval_pipeline")
         return {"response": "No relevant information found."}


# In[6]:


def agent1(state):
    print("entering agent 1")
    text = state["messages"][-1].content
    print(f"Received message: {text}")
 
    category = classify_sentiment(text)
    print(f"Classified as: {category}")
    print("coming out of agent 1")
    return {"category": category}


# In[7]:


def agent2(state):
    print("entering agent 2")
    query_text = state["messages"][-1].content
    retrieved_docs = query_retrieval_pipeline(state)
    print("coming out of agent 2")
    print(f"Retrieved documents: {retrieved_docs}")
    response=generate_llm_response(query_text,retrieved_docs)
    return {"response": response}


# In[8]:


def classify(state):
    print("entering classify")
    print(f"Classifying category: {state['category']}")
 
    if state["category"] == "Query":
        print("exiting classify")
        return {"response": "Proceeding to Agent 2..."}
    else:
        print("exiting classify")
        return {"response": "Please contact customer support."}


# In[9]:


def generate_llm_response(query, context):
    start_time= time.time()
    API_URL = "https://api-inference.huggingface.co/models/google/gemma-2-2b-it"
    headers = {"Authorization": f"Bearer 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"}
    messages = [
        {
            "role": "system",
            "content": "You are an AI assistant that provides precise, concise, and accurate responses. "
            "Use the provided context to answer the user's query accurately."
            "If the context does not contain sufficient information, state that explicitly",
        },
        {
            "role": "user",
             "content": f"Context: {context}\n\nQuery: {query}"
             # f"Provide a detailed yet concise answer to the {query} using only the information from the {context}. "
             "If relevant details are missing, mention that additional information is required."
        },
    ]


    formatted_input = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages])
    Payload = {
    "inputs": formatted_input,
    "parameters": {
        "temperature": 0.7
    }
}
    response = requests.post(API_URL, headers=headers, json=Payload)
    end_time=time.time()
    execution= end_time - start_time
    print(execution,"seconds")
    print(response)
    return response.json()
    


# In[10]:


def agent_logic(query):
    category=classify_sentiment(query)
    if category.lower() == "complaint":
        return {"response": "Thank you for your feedback. Your complaint has been recorded, and our team will review it shortly."}
    elif category.lower() == "query":
        retrieved_docs = query_retrieval_pipeline(query)
        response = generate_llm_response( query,retrieved_docs)
        return {"response": response}
    else:
        return {"response": "I'm not sure how to handle this request. Please try rephrasing."}


# In[11]:


class State(TypedDict):
    messages: Annotated[list, add_messages]
    category: str
    response: str
 
graph_builder = StateGraph(State)


# In[12]:


graph_builder.add_node("Agent 1", agent1)
graph_builder.add_node("Classify", classify)
graph_builder.add_node("Agent 2", agent2)
graph_builder.add_conditional_edges(
    "Agent 1",
    lambda state: "Agent 2" if state["category"] == "Query" else "Classify"
)


# In[13]:


graph_builder.add_edge(START, "Agent 1")
graph_builder.add_edge("Agent 2", END)
graph_builder.add_edge("Classify", END)


# In[14]:


graph = graph_builder.compile()


# In[15]:


from IPython.display import Image, display
 
try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    pass


# In[ ]:


# while True:
#     user_input = input("User: ")
#     if user_input.lower() == "quit":
#         break
#     events = list(graph.stream({"messages": [{"role": "user", "content": user_input}]}))
#     print(events)
#     if not events:
#         print("No events generated")
#         continue       
#     for event in events:
#         for key, value in event.items():
#             if key == 'Agent 1':
#                 category = value['category']
#                 print("Category:", category)

#             if category=="Complaint":
#                 if key == 'Classify':
#                     response = value['response']
#                     print("Assistant:", response)
#             else:
#                 print("Entering else part")
#                 if "response" in value:
#                     print("Entering else part2")
#                     text=value["response"]
                    # generated_text =text[0]["generated_text"]
                    # answer_start = generated_text.find("Answer:")
                    # print(answer_start)
                    # if answer_start != -1:
                    #      answer = text[answer_start + len("Answer:"):].strip()
                    # else:
                    #     answer = "Answer not found in response."
                    #     print("Assistant",answer)
while True:
    user_input = input("User: ")
    if user_input.lower() == "quit":
        break
    events = list(graph.stream({"messages": [{"role": "user", "content": user_input}]}))
    print(events)
    if not events:
        print("No events generated")
        continue       
    for event in events:
        for key, value in event.items():
            if key == 'Agent 1':
                category = value['category']
                print("Category:", category)

            if category == "Complaint":
                if key == 'Classify':
                    response = value['response']
                    print("Assistant:", response)
            else:
                print("Entering else part")
                if "response" in value:
                    print("Entering else part2")
                    text = value["response"]
                    generated_text = text[0]["generated_text"]
                    answer_start = generated_text.find("Answer:")
                    print(answer_start)
                    if answer_start != -1:
                        answer = generated_text[answer_start + len("Answer:"):].strip()
                        print("Assistant:", answer)
                    else:
                        answer = "Answer not found in response."
                        print("Assistant:", answer)

                
                
        
        # for key, value in event.items():
        #     print(key, ":", value)
        # if "response" in value:
        #      text=value["response"]
        #      if isinstance(text, list) and text and isinstance(text[0], dict) and "generated_text" in text[0]:
        #          generated_text = text[0]["generated_text"]
        #          if "Answer:" in generated_text:
        #            answer_start = generated_text.find("Answer:")
        #            answer = generated_text[answer_start + len("Answer:"):].strip()
        #          else:
        #            answer = "Answer not found in response."
        #      elif isinstance(text, str):
        #          if "Answer:" in text:
        #              answer_start = text.find("Answer:")
        #              answer = text[answer_start + len("Answer:"):].strip()   
        #          else:
        #              answer = "Answer not found in response."
        #      else:
        #          answer = "Invalid response format."
        #      print(answer)
             # generated_text =text[0]["generated_text"]
             # answer_start = generated_text.find("Answer:")
            

    


# In[ ]:




