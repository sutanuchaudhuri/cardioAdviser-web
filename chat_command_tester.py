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
from  constants import corpus_reference as corpusRef
import os
import pickle
from langchain_core.messages import HumanMessage
# Load environment variables
load_dotenv()
# openai_api_key = os.getenv("OPENAI_API_KEY")
# Initialize OpenAI
# openai = OpenAI(
#     api_key = openai_api_key
# )
llm = ChatOpenAI(model="gpt-4o-mini",streaming=True)

chat_responses = []
chat_history = []






system_prompt=" You are an assistant for bypass surgery related questions including symptoms. "\
    ""\
    " If you don't know the answer, say that you "\
    "don't know. Use twenty sentences maximum. "\
    "Include any hyperlink for credible sources. Refer to hospital urls in internet as needed"\
    "\n\n"

# result = chain({"question": question+ system_prompt, "chat_history": chat_history})
# chat_history.append((question, result['answer']))
# return result['answer']

texts=None
with open('document.pickle', 'rb') as handle:
    texts = pickle.load(handle)

doc = Document(page_content=texts[0])
vectorstore = Chroma.from_documents(documents=[doc], embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()
# Initialize ConversationalRetrievalChain with streaming enabled
chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)

response = chain.invoke({"question": 'Hi how r u?', "chat_history": chat_history})
print(response)

# @app.websocket("/ws")
# async def chat(websocket: WebSocket):
#
#     await websocket.accept()
#     while True:
#         user_input = await websocket.receive_text()
#         chat_responses.append(user_input)
#
#         try:
#             response = chain.invoke({"question": user_input,"chat_history": chat_history}) #"chat_history": chat_log
#             ai_response = ''
#
#             for chunk in response:
#                 if chunk is not None:
#                     ai_response += chunk
#                     await websocket.send_text(chunk)
#             chat_responses.append(ai_response)
#             chat_history.extend([HumanMessage(content=user_input), ai_response])
#
#         except Exception as e:
#             await websocket.send_text(f'Error: {str(e)}')
#             break
#
#
#
#
#
#
# @app.post("/", response_class=HTMLResponse)
# async def chat(request: Request, user_input: Annotated[str, Form()]):
#
#
#
#     response = chain.invoke({"question": user_input,"chat_history": chat_history}) # "chat_history": chat_log
#     ai_response = response
#     #bot_response=response
#     chat_history.extend([HumanMessage(content=user_input), ai_response])
#     chat_responses.append(ai_response)
#
#     return templates.TemplateResponse("home.html", {"request": request, "chat_responses": chat_responses})
#
#
#
#
#
#
#
#
#
#












