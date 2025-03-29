import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessageChunk
from langchain_core.documents import Document
import pickle
# from logging import logging
from bs4  import BeautifulSoup
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from dotenv import load_dotenv, find_dotenv
import os

from sqlalchemy.testing.suite.test_reflection import metadata

if os.environ.get("DYNO") is None: #Check if running on Heroku
    load_dotenv(find_dotenv())



# # Load, chunk and index the contents of the blog.
# bs_strainer = bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))
# loader = WebBaseLoader(
#     web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
#     bs_kwargs={"parse_only": bs_strainer},
# )
# docs = loader.load()
#
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# splits = text_splitter.split_documents(docs)
# vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
#
# # Retrieve and generate using the relevant snippets of the blog.
# retriever = vectorstore.as_retriever().with_config(
#     tags=["retriever"]
# )
# We need to add streaming=True

def extract_html_text(html_content_file_name,reference_url):
    soup = BeautifulSoup(open(f"resources/{html_content_file_name}", encoding="utf8"), "html.parser")
    html_text= soup.get_text(separator=" ", strip=True)
    document=Document(page_content=html_text, metadata={'source': f'{reference_url}'})
    return document



def load_document_from_web(urls):

    documents=[]
    for url in urls:
        loader = WebBaseLoader(
            url
        )
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        for split in splits:
            documents.append(split)

    return documents

html_docs={"a.html":"https://www.pennmedicine.org/make-an-appointment"
           ,"b.html":"https://www.pennmedicine.org/providers?keyword=Penn-Heart-Surgery-Program&keywordid=57534&keywordtypeid=11"
           ,"c.html":"https://www.hopkinsmedicine.org/heart-vascular-institute/cardiac-surgery"
           # ,"d.html":"https://www.hopkinsmedicine.org/heart-vascular-institute/cardiac-surgery/mitral-valve-repair-replacement"
           }


# Define the URLs to load
urls = [
    "https://diwakaraditi30.wixsite.com/cardiac-help/general-info-cabg"
    ,"https://diwakaraditi30.wixsite.com/cardiac-help/causes"
    ,"https://diwakaraditi30.wixsite.com/cardiac-help/risks"
    ,"https://diwakaraditi30.wixsite.com/cardiac-help/non-surgical-alternatives"
    ,'https://www.pennmedicine.org/for-patients-and-visitors/find-a-program-or-service/heart-and-vascular/heart-surgery'
    ,'https://www.pennmedicine.org/make-an-appointment'
    ,'https://www.pennmedicine.org/providers?keyword=Penn-Heart-Surgery-Program&keywordid=57534&keywordtypeid=11'
    ,'https://www.hopkinsmedicine.org/heart-vascular-institute/cardiac-surgery'
    ,'https://www.hopkinsmedicine.org/heart-vascular-institute/cardiac-surgery/mitral-valve-repair-replacement'

]

#load documents from web
documents=load_document_from_web(urls)

#load document from static html
for html_doc_key in html_docs:
   documents.append(extract_html_text(html_content_file_name=html_doc_key,reference_url=html_docs[html_doc_key]))

#load document from pickle
texts=None
with open('document.pickle', 'rb') as handle:
    texts = pickle.load(handle)

aditi_doc = Document(page_content=texts[0], metadata={'source':'Author: Aditi Diwakar, '
                                                               'Title: Sugery- Patient Guide'})
documents.append(aditi_doc)



# Print the loaded documents
for doc in documents:
    print(doc)



#RAG chain component STARTS here


vectorstore = Chroma.from_documents(documents=documents, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever().with_config(
    tags=["retriever"]
)



llm = ChatOpenAI(model="gpt-4o-mini",temperature=0,streaming=True)
#llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, streaming=True)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
contextualize_q_chain = (contextualize_q_prompt | llm | StrOutputParser()).with_config(
    tags=["contextualize_q_chain"]
)

qa_system_prompt = """You are an assistant for question-answering tasks related to bypass surgery. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use ten sentences maximum.\
\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


def contextualized_question(input: dict):
    if input.get("chat_history"):
        return contextualize_q_chain
    else:
        return input["question"]


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
        RunnablePassthrough.assign(context=contextualize_q_chain | retriever | format_docs)
        | qa_prompt
        | llm
).with_config(
    tags=["main_chain"]
)

app = FastAPI()

# Allow CORS for all origins (for testing purposes; restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chat_responses = []
@app.get("/")
async def root():
    return FileResponse("static/index.html")


def serialize_aimessagechunk(chunk):
    """
    Custom serializer for AIMessageChunk objects.
    Convert the AIMessageChunk object to a serializable format.
    """
    if isinstance(chunk, AIMessageChunk):
        return chunk.content
    else:
        raise TypeError(
            f"Object of type {type(chunk).__name__} is not correctly formatted for serialization"
        )


async def generate_chat_events(message):

    print("===================== START Chat model has started. generate_chat_events ====================")
    print(message)
    print("=====================END Chat model has started. generate_chat_events ====================")
    final_message = ""
    try:
        async for event in rag_chain.astream_events(message, version="v1"):
            # Only get the answer
            sources_tags = ['seq:step:3', 'main_chain']
            if all(value in event["tags"] for value in sources_tags) and event["event"] == "on_chat_model_stream":
                chunk_content = serialize_aimessagechunk(event["data"]["chunk"])
                if len(chunk_content) != 0:
                    final_message += chunk_content
                    data_dict = {"data": chunk_content}
                    data_json = json.dumps(data_dict)
                    yield f"data: {data_json}\n\n"

            # Get the reformulated question
            sources_tags = ['seq:step:2', 'main_chain', 'contextualize_q_chain']
            if all(value in event["tags"] for value in sources_tags) and event["event"] == "on_chat_model_stream":
                chunk_content = serialize_aimessagechunk(event["data"]["chunk"])
                if len(chunk_content) != 0:
                    data_dict = {"reformulated": chunk_content}
                    data_json = json.dumps(data_dict)
                    yield f"data: {data_json}\n\n"

            # Get the context
            sources_tags = ['main_chain', 'retriever']
            if all(value in event["tags"] for value in sources_tags) and event["event"] == "on_retriever_end":
                documents = event['data']['output']['documents']
                # Create a new list to contain the formatted documents
                formatted_documents = []
                # Iterate over each document in the original list
                for doc in documents:
                    # Create a new dictionary for each document with the required format
                    formatted_doc = {
                        'page_content': doc.page_content,
                        'metadata': {
                            'source': doc.metadata['source'],
                        },
                        'type': 'Document'
                    }
                    # Add the formatted document to the final list
                    formatted_documents.append(formatted_doc)

                # Create the final dictionary with the key "context"
                final_output = {'context': formatted_documents}

                # Convert the dictionary to a JSON string
                data_json = json.dumps(final_output)


                yield f"data: {data_json}\n\n"
            if event["event"] == "on_chat_model_end":
                print("Chat model has completed one response.")
                if final_message != "":
                    print("====================================FINAL output==============================")
                    print("Final message:", final_message)
                    print("====================================END FINAL output==============================")


    except Exception as e:
        print('error' + str(e))


@app.get("/chat_stream/{message}")
async def chat_stream_events(message: str):
    return StreamingResponse(generate_chat_events({"question": message, "chat_history": []}),
                             media_type="text/event-stream")


# if __name__ == "__main__":
#     import uvicorn
#
#     uvicorn.run(app, host="0.0.0.0", port=8000)