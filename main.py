import re 
from typing import List
import urllib.parse
import requests
import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import chromadb
import time
from IPython.display import Markdown
import fitz
from youtube_transcript_api import YouTubeTranscriptApi
from chromadb.utils import embedding_functions
from chromadb import Documents, EmbeddingFunction, Embeddings
from tqdm import tqdm
from flask import Flask, render_template, send_file, redirect, url_for, request, session
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

def init_db():
    global db
    db = get_or_create_collection('sme')

app = Flask(__name__)
app.secret_key = 'your_secret_key'


def extract_text_from_pdf(pdf_path):
    try:
        with fitz.open(pdf_path) as pdf_document:
            text = ""
            for page_number in range(len(pdf_document)):
                page = pdf_document.load_page(page_number)
                page_text = page.get_text()
                text += page_text
            return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None
    
def create_content_embeddings_db(data: str, db_name: str) -> chromadb.Collection:
    docs = chunking(data)
    embedding_function = sentence_transformer_ef
    embeddings = embedding_function(docs)
    db = create_chroma_db(docs, embeddings, db_name)
    return db

def chunking(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True
    )
    texts = text_splitter.create_documents([data])
    docs = [content for sublist in texts for content in sublist]
    return docs
   
def create_chroma_db(docs: List, embeddings: Embeddings, name) -> chromadb.Collection:
    class MyEmbeddingFunction(EmbeddingFunction):
        def __call__(self, input: Documents) -> Embeddings:
            embeddings_list = sentence_transformer_ef([docs])
            embeddings = Embeddings(embeddings_list)
            return embeddings
    chroma_client = chromadb.PersistentClient(path="D:/Mini Projects/TubeChat/database")
    db = chroma_client.get_or_create_collection(name=name, embedding_function=sentence_transformer_ef)
    
    initial_size = db.count()
    for i, (doc, emb) in tqdm(enumerate(zip(docs, embeddings)), total=len(docs), desc="Creating Chroma DB"):
        db.add(documents=doc,embeddings=emb,ids=str(i + initial_size))
        time.sleep(0.5)
    return db

def get_or_create_collection(name: str) -> chromadb.Collection:
    chroma_client = chromadb.PersistentClient(path="D:/Mini Projects/TubeChat/database")
    try:
        return chroma_client.get_collection(name=name)
    except ValueError:
        # If the collection does not exist, create it with a default embedding function
        return chroma_client.create_collection(name=name, embedding_function=EmbeddingFunction())


def list_to_string(passages):
    content = ""
    for passage in passages:
        content += passage + "\n"
    return content

def get_relevant_passages(query):
    if db is None:
        raise ValueError("Database not initialized")
    passages = db.query(query_texts=[query], n_results=5)['documents'][0]
    return passages

def ollama_llm(question, context):
    formatted_prompt = f'''
    Question: {question}Context: {context}
    Answer the question in 1250-2500 words. If asked explain code. Keep a smooth flow for your answer. If the question is asking for a code then also explain the algorithm.Incase of spelling errors in the context, make spelling corrections in the response.
    If the information is not sufficient then give output as "Info not good to answer".
    '''
    response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': formatted_prompt}])
    return response['message']['content']

def get_answer(ques):
    passages = get_relevant_passages(ques)
    answer = ollama_llm(ques,passages)
    return answer



def get_video_id(video_url):
    match = re.search(r"(?<=v=)[\w-]+", video_url)
    return match.group(0) if match else None

def get_video_transcript(video_url):
    video_id = get_video_id(video_url)
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = ""
        for line in transcript:
            text += line['text'] + " "
        return text.strip()
    except Exception as e:
        print(f"Error getting transcript: {e}")
        return None


@app.route('/')
def start():
    return render_template('index.html')

@app.route('/form')
def start_pdf():
    return render_template('pdf.html')

@app.route('/get_pdf', methods = ['GET', 'POST'])
def upload_pdf():
    if request.method == 'GET':
        if 'file' not in request.files:
            return render_template('pdf.html', error='No file uploaded.')
        file = request.files['file']
        if file.filename.endswith('.pdf'):
            file_path = 'uploaded_file.pdf'
            file.save(file_path)
            data = extract_text_from_pdf(file_path)
            db = create_content_embeddings_db(data,'pdf_db')
    return render_template('question_pdf.html')
    # return render_template('pdf.html')
        
@app.route('/get_answer', methods=['GET', 'POST'])
def upload_answer():
    answer = "No question provided."  
    if request.method == 'POST':
        question = request.form.get('question', '')
        if question:  
            try:
                answer = get_answer(question)
            except Exception as e:
                answer = f"An error occurred while finding the answer: {str(e)}"
        else:
            answer = "Please submit a valid question."
    return render_template('output_pdf.html', answer=answer)


@app.route('/youtube')
def start_yt():
    return render_template('yt.html')

@app.route('/get_url', methods=['GET', 'POST'])
def upload_url():
    if request.method == 'POST':
        youtube_url = request.form['url']
        text = get_video_transcript(youtube_url)
        db = create_content_embeddings_db(text,'youtube_db')
    return render_template('question_yt.html')
 
@app.route('/get_yt_answer', methods=['GET', 'POST'])
def upload_yt_answer():
    if request.method == 'POST':
        ques = request.form['question']
        answer = get_answer(ques)
    return render_template('output_yt.html',answer=answer)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)

