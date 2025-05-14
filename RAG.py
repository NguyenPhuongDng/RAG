# Necessary inputs
from FlagEmbedding import FlagReranker
import os
import time
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, format_document
from langchain.prompts.prompt import PromptTemplate
import requests
import sys
from langchain_core.prompts import ChatPromptTemplate, format_document
from langchain.prompts.prompt import PromptTemplate

os.environ["TOGETHER_API_KEY"] = "414229540a05a7ce253fd2bfc33d221a19ff3277c6bd7bde1056eca4bc0f18ae"  # Removed the API Key that we used

model_name = "BAAI/bge-large-en-v1.5"
encode_kwargs = {'normalize_embeddings': True} 

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_norm = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={'device': DEVICE},
    encode_kwargs=encode_kwargs)

db3 = Chroma(persist_directory="db", embedding_function=model_norm)

retriever = db3.as_retriever(search_kwargs={"k": 100})
def get_answer(input_text) -> str:
    url = 'https://api.together.xyz/inference'
    headers = {
      'Authorization': 'Bearer ' + os.environ["TOGETHER_API_KEY"],
      'accept': 'application/json',
      'content-type': 'application/json'
    }
    time.sleep(12)  # Nghỉ 12 giây để tránh rate limit

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
    print(response.status_code)
    print(response.json())

    text = response.json()['output']['choices'][0]['text']
    print(text)
    return text

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

def combine_documents(docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

def combine_documents_2(docs, document_separator="\n\n"):
    combined_docs = []
    for doc in docs:
        combined_docs.append(doc[1])
    return document_separator.join(combined_docs)

def split_questions(string, delimiter='?'):
    questions = string.split(delimiter)
    questions = [question.strip().split(". ", 1)[-1] + delimiter for question in questions]
    return questions

# Sử dụng Retriever (thường)
def _unique_documents(documents):
    return [doc for i, doc in enumerate(documents) if doc not in documents[:i]]
with open('test.csv', 'r') as file:
    questions = file.readlines()
for question in questions:
    docs = retriever.invoke(question)

    prompt_start = """
    You are an assistant for question-answering tasks and the questions are related to the city of Pittsburgh. Use the following pieces of retrieved context to answer the question. Do not exceed one sentence for the answer. Do not be verbose when generating the answer. Give out the answer directly even if it does not form a coherent sentence.

    """
    context = combine_documents(docs[:3])
    input_text = prompt_start + "Question: " + question + "Context: " + context + "Answer: "
    time.sleep(1)
    print(question)

    answer = get_answer(input_text)
    with open('rag_system_answers.txt', 'a', encoding='utf-8') as output_file:
        output_file.write(f'{answer}\n==================\n')

# Sử dụng Retriever + Reranking        
reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)

with open('test.csv', 'r', encoding = 'utf-8') as file:
    questions = file.readlines()
for question in questions:
    docs = retriever.invoke(question)

    prompt_start = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. The questions are about the city of Pittsburgh. Do not exceed one sentence for the answer. Do not be verbose when generating the answer. Provide the most direct and concise answer even if it lacks grammatical correctness.
    """
    docs_to_rerank = []
    for i in range(len(docs)):
        docs_to_rerank.append([question, str(docs[i])])
    scores = reranker.compute_score(docs_to_rerank)

    combined_data = list(zip(docs_to_rerank, scores))
    sorted_data = sorted(combined_data, key=lambda x: x[1], reverse=True)
    sorted_docs_to_rerank, sorted_scores = zip(*sorted_data)
    top_k_docs = sorted_docs_to_rerank[:3]
    context = combine_documents_2(top_k_docs)

    input_text = prompt_start + "Question: " + question + "Context: " + context + "Answer: "
    print(question)
    time.sleep(1)

    answer = get_answer(input_text)
    print("=====================")
    with open('rag_with_reranker_answers.txt', 'a', encoding = 'utf-8') as output_file:
        output_file.write(f'{answer}\n==================\n')


# Sử dụng Retriever + Reranking + Multi query retriever
delimiter = '?'

with open('test.csv', 'r') as file:
    questions = file.readlines()
    
for question in questions:
    
    input_text_for_ques = f"""
    [TASK]: Write the below question in 3 different ways.
    [QUESTION]: {question}
    """
    
    diff_questions = get_answer(input_text_for_ques)
    
    paraphrased_ques = split_questions(diff_questions, delimiter)
    paraphrased_ques = paraphrased_ques[:-1]
    print(paraphrased_ques)
    
    all_docs = []
    for single_question in paraphrased_ques:
        all_docs.extend(retriever.get_relevant_documents(single_question))
        
    all_docs.extend(retriever.get_relevant_documents(question))
    
    unique_docs = _unique_documents(all_docs)

    prompt_start = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. Do not exceed one sentence for the answer. Do not be verbose when generating the answer. Give out the answer directly even if it does not form a coherent sentence.
    """
    
    docs_to_rerank = []
    for i in range(len(unique_docs)):
        docs_to_rerank.append([question, str(unique_docs[i])])
    scores = reranker.compute_score(docs_to_rerank)

    combined_data = list(zip(docs_to_rerank, scores))
    sorted_data = sorted(combined_data, key=lambda x: x[1], reverse=True)
    sorted_docs_to_rerank, sorted_scores = zip(*sorted_data)
    top_k_docs = sorted_docs_to_rerank[:3]
    context = combine_documents_2(top_k_docs)

    input_text = prompt_start + "Question: " + question + "Context: " + context + "Answer: "
    print(question)
    time.sleep(1)

    answer = get_answer(input_text)
    print("=====================")
    with open('rag_with_reranker_with_multiquery_answers.txt', 'a', encoding = 'utf-8') as output_file:
        output_file.write(f'{answer}\n==================\n')


# Sử dụng Retriever + Reranking + Multi query retriever + Few-shot prompting
delimiter = '?'

with open('test.csv', 'r') as file:
    questions = file.readlines()
    
for question in questions:
    input_text_for_ques = f"""
    [TASK]: Rewrite the following question in three distinct formulations.
    [ORIGINAL QUESTION]: {question}
    """
    diff_questions = get_answer(input_text_for_ques)
    
    paraphrased_ques = split_questions(diff_questions, delimiter)
    paraphrased_ques = paraphrased_ques[:-1]
    print(paraphrased_ques)
    all_docs = []
    for single_question in paraphrased_ques:
        all_docs.extend(retriever.get_relevant_documents(single_question))
    all_docs.extend(retriever.get_relevant_documents(question))
    unique_docs = _unique_documents(all_docs)
    prompt_start = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. The questions are about Pittsburgh. Provide only the factual answer in the briefest form possible, even if it lacks full grammar. Avoid redundant or explanatory sentences.

    Here are some examples to illustrate the answer format:

    Query: What is the population of Pittsburgh?
    Answer: ~300,000

    Query: Famous sports team in Pittsburgh?
    Answer: Pittsburgh Steelers

    Query: What river runs through Pittsburgh?
    Answer: Allegheny River
    """
    docs_to_rerank = []
    for i in range(len(unique_docs)):
        docs_to_rerank.append([question, str(unique_docs[i])])
    scores = reranker.compute_score(docs_to_rerank)

    combined_data = list(zip(docs_to_rerank, scores))
    sorted_data = sorted(combined_data, key=lambda x: x[1], reverse=True)
    sorted_docs_to_rerank, sorted_scores = zip(*sorted_data)
    top_k_docs = sorted_docs_to_rerank[:3]
    context = combine_documents_2(top_k_docs)

    input_text = prompt_start + "Question: " + question + "Context: " + context + "Answer: "
    print(question)
    time.sleep(1)
    answer = get_answer(input_text)
    print("=====================")
    with open('rag_with_reranker_with_multiquery_with_few_shot_answers.txt', 'a', encoding = 'utf-8') as output_file:
        output_file.write(f'{answer}\n==================\n')
