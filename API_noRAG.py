import os
import time
import torch
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.prompts import ChatPromptTemplate, format_document
from langchain.prompts.prompt import PromptTemplate
import requests
import sys
from langchain_core.prompts import ChatPromptTemplate, format_document
from langchain.prompts.prompt import PromptTemplate

# Setting the Together API Key
os.environ["TOGETHER_API_KEY"] = "414229540a05a7ce253fd2bfc33d221a19ff3277c6bd7bde1056eca4bc0f18ae"  

def get_answer(input_text) -> str:
    url = 'https://api.together.xyz/inference'
    headers = {
      'Authorization': 'Bearer ' + os.environ["TOGETHER_API_KEY"],
      'accept': 'application/json',
      'content-type': 'application/json'
    }
    data = {
      "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
      "prompt": input_text,
      "max_tokens": 100,
      "temperature": 0.7,
      "top_p": 0.7,
      "top_k": 50,
      "repetition_penalty": 1
    }
    response = requests.post(url, json=data, headers=headers)
    #print(response.json())
    text = response.json()['output']['choices'][0]['text']
    print(text)
    return text

# Inference

with open('test/questions.txt', 'r', encoding='utf-8') as file:
    questions = file.readlines()
for question in questions:

    prompt_start = """
    You are an assistant for question-answering tasks and the questions are related to the University of Pittsburgh. Do not exceed one sentence for the answer. Do not be verbose when generating the answer. Give out the answer directly even if it does not form a coherent sentence."""
    input_text = prompt_start + "Question: " + question + "Answer: "
    time.sleep(12)
    print(question)
    answer = get_answer(input_text)
    with open('test/answers.txt', 'a', encoding='utf-8') as output_file:
        output_file.write(f'{answer}\n')