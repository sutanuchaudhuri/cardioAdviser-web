from openai import OpenAI
from fastapi import FastAPI, Form, Request, WebSocket
from typing import Annotated
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import os
from dotenv import load_dotenv
import getpass
import os
import gradio as gr
import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from PyPDF2 import PdfReader
from langchain.chains import ConversationalRetrievalChain
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import YoutubeLoader


from openai import OpenAI
import requests
from langchain_openai import ChatOpenAI
from sqlalchemy.testing.suite.test_reflection import metadata

from  constants import corpus_reference as corpusRef
import os

def processYoutubeUrls():
    youtube_urls=corpusRef.YOUTUBE_URLs
    documents = []

    for idx, url in enumerate(youtube_urls):
        print(f" processing URL {str(idx + 1)} : {url}")
        try:
            loader = YoutubeLoader.from_youtube_url(
                url,
                add_video_info=False)
            doc = loader.load()[0]
            documents.append(doc)
        except Exception as e:
            print(f"Error processing URL {url}: {e}")
    return documents






def processPdfDocument(file_path):
    pdf_file = PdfReader(file_path)
    text_data = ''
    for pg in pdf_file.pages:
        text_data += pg.extract_text()
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
         chunk_size=200, chunk_overlap=20
    )
    texts = text_splitter.split_text(text_data)
    main_doc = Document(page_content=texts[0],metadata="Key research document")
    return main_doc

def getDocumentList():
    file_path='Health-Cardio.pdf'
    documents=[]
    # documents=processYoutubeUrls1()
    document=processPdfDocument(file_path)
    documents.append(document)
    return documents

documents=getDocumentList()