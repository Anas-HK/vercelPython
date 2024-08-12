import fitz  # PyMuPDF
import re
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from http.server import BaseHTTPRequestHandler
from urllib import parse

# Set your OpenAI API key
openai.api_key = "sk-proj-ul44Q4GsK0DM09LKRqVHT3BlbkFJNfK1KgPsggeyzp2Zdyya"

# Define PDF paths relative to the api directory
pdf_paths = [
    './api/rainenotfunc.pdf',
    './api/Email_consent_form.pdf'
]

def extract_text_from_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    text_by_page = []
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text = page.get_text()
        text_by_page.append(text)
    return text_by_page

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_embeddings(texts, model="text-embedding-ada-002"):
    response = openai.Embedding.create(
        input=texts,
        model=model
    )
    embeddings = [item['embedding'] for item in response['data']]
    return embeddings

def index_documents(pdf_paths):
    all_texts = []
    all_embeddings = []
    
    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        text_by_page = extract_text_from_pdf(pdf_path)
        preprocessed_texts = [preprocess_text(text) for text in text_by_page]
        embeddings = get_embeddings(preprocessed_texts)
        
        all_texts.extend(preprocessed_texts)
        all_embeddings.extend(embeddings)
    
    return all_texts, all_embeddings

def retrieve_relevant_documents(query, texts, embeddings, model="text-embedding-ada-002", top_k=5):
    query_embedding = get_embeddings([query], model=model)[0]
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    top_k_indices = similarities.argsort()[-top_k:][::-1]
    relevant_texts = [texts[i] for i in top_k_indices]
    return relevant_texts

def generate_response(query, relevant_texts, model="gpt-3.5-turbo"):
    prompt = f"Query: {query}\n\nRelevant Information:\n"
    for text in relevant_texts:
        prompt += f"- {text}\n"
    prompt += "\nResponse:"
    
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Also please format the result in html so that it formats correctly as I need to send it in an email. Also be under max 250 tokens."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=250
    )
    return response['choices'][0]['message']['content'].strip()

class handler(BaseHTTPRequestHandler):

    def do_GET(self):
        s = self.path
        dic = dict(parse.parse_qsl(parse.urlsplit(s).query))
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()

        if "query" in dic:
            query = dic["query"]

            # Index documents from predefined PDFs
            texts, embeddings = index_documents(pdf_paths)
            
            # Retrieve relevant documents
            relevant_texts = retrieve_relevant_documents(query, texts, embeddings)
            
            # Generate response
            response = generate_response(query, relevant_texts)
        else:
            response = "Please provide a query parameter."

        self.wfile.write(response.encode())
        return
