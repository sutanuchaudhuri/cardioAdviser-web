import json
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessageChunk
from reportlab.pdfbase import pdfmetrics
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
from starlette.middleware.sessions import SessionMiddleware
from PyPDF2 import PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from io import BytesIO
from reportlab.lib import colors
import re
if os.environ.get("DYNO") is None: #Check if running on Heroku
    load_dotenv(find_dotenv())
pdfmetrics.registerFont(TTFont('NotoSansDevanagari', 'static/NotoSansDevanagari-Regular.ttf'))


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
    "https://www.hopkinsmedicine.org/health/treatment-tests-and-therapies/heart-transplant"
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
urls_doc=[
    "https://www.lung.org/lung-health-diseases/lung-procedures-and-tests/lobectomy#:~:text=A%20lobectomy%20is%20a%20surgical,lobes%20of%20your%20left%20lung.",
    "https://my.clevelandclinic.org/health/treatments/17608-lobectomy#procedure-details",
    "https://www.hopkinsmedicine.org/health/treatment-tests-and-therapies/lobectomy",
    "https://ctsurgerypatients.org/procedures/lobectomy",
    "https://www.cancercenter.com/cancer-types/lung-cancer/diagnosis-and-detection/lobectomy",
    "https://siteman.wustl.edu/treatment/cancer-types/lung-cancer/treatments/robotic-assisted-lobectomy/",
    "https://www.thoracicsurgery.co.uk/lobectomy/",
    "https://stanfordhealthcare.org/medical-treatments/v/vats/vats-types/vats-lobectomy.html",
    "https://www.svhlunghealth.com.au/procedures/procedures-treatments/video-assisted-thoracoscopic-surgery-vats-lobectomy",
    "https://www.beaumont.org/treatments/lobectomy",
    "https://www.who.int/news-room/fact-sheets/detail/tuberculosis#:~:text=Tuberculosis%20(TB)%20is%20an%20infectious,been%20infected%20with%20TB%20bacteria.",
    "https://www.hopkinsmedicine.org/health/treatment-tests-and-therapies/lobectomy#:~:text=Fungal%20infection.%20Fungi%20can%20grow%20in%20the%20body%20and%20cause%20infections.",
    "https://www.cancer.org/cancer/types/lung-cancer/about/what-is.html",
    "https://thesurgicalclinics.com/lung-surgery-why-might-someone-need-a-lobectomy/",
    "https://www.bswhealth.com/treatments-and-procedures/lobectomy",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC4543327/",
    "https://publications.ersnet.org/content/erj/46/suppl59/pa2202",
    "https://www.cancerresearchuk.org/about-cancer/lung-cancer/treatment/non-small-cell-lung-cancer",
    "https://nyulangone.org/conditions/chronic-obstructive-pulmonary-disease/treatments/lifestyle-changes-for-chronic-obstructive-pulmonary-disease",
    "https://resources.healthgrades.com/right-care/lungs-breathing-and-respiration/lobectomy",
    "https://healthinfo.coxhealth.com/Conditions/Cancer/92,P07749",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC4606870/#:~:text=A%20recent%20review%20of,lobectomy%20%5B3%5D.&text=is%2010%25%20to%2050%25%2C,lobectomy%20%5B3%5D.&text=the%20operations%20is%20increasing,lobectomy%20%5B3%5D.&text=The%20most%20common%20complications,lobectomy%20%5B3%5D.",
    "https://www.verywellhealth.com/lobectomy-complications-and-prognosis-2249329",
    "https://www.sciencedirect.com/science/article/abs/pii/S0003497522003745",
    "https://shc.amegroups.org/article/view/5996/html",
    "https://lungfoundation.com.au/blog/understanding-lobectomies/",
    "https://academic.oup.com/ejcts/article/20/4/694/373980",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC6462550/#:~:text=found%20median%20cost%20of,registry%20%2815%2C18%29.",
    "https://atlanticendomd.com/how-much-does-thyroid-surgery-cost-with-insurance/#:~:text=Understanding%20Different%20Thyroid%20Surgeries,Minimally%20invasive",
    "https://www.annalsthoracicsurgery.org/article/S0003-4975(20)31896-8/fulltext",
    "https://myhealth.alberta.ca/Health/aftercareinformation/pages/conditions.aspx?hwid=zy1364",
    "https://med.uth.edu/cvs/patient-care/conditionsandprocedures/video-assisted-thoracic-surgery-vats/recovery-after-vats-and-robotic-lobectomy-surgery/",
    "https://www.mountsinai.org/health-library/discharge-instructions/lung-surgery-discharge#:~:text=Ask%20your%20surgeon%20when,is%20heavy.",
    "https://www.henryford.com/blog/2021/11/lungs-after-lobectomy",
    "https://go2.org/resources-and-support/support-groups/",
    "https://www.mskcc.org/cancer-care/patient-education/after-your-thoracic-surgery#:~:text=You%27ll%20have%20less%20pain,and%20discomfort.",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC4367696/",
    "https://www.sciencedirect.com/science/article/pii/S0022522320300544",
    "https://tlcr.amegroups.org/article/view/2053/html",
    "https://acsjournals.onlinelibrary.wiley.com/doi/10.1002/cncr.24625",
    "https://pubmed.ncbi.nlm.nih.gov/32067786/#:~:text=Results%3A%20Among%20the%20543,The%2010%2Dyear",
    "https://ascopubs.org/doi/10.1200/JCO.2011.39.5269#:~:text=Treatment%20rates%20decreased%20more,lower%20rates",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC4314218/#:~:text=Women%20with%20lung%20cancer,on%20sex.",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC5812758/#:~:text=Black%20patients%20with%20lung,lung%20cancer.",
    "https://www.mylungcancerteam.com/resources/lobectomy-for-lung-cancer-how-to-prepare-and-recover#:~:text=Your%20doctor%20may%20advise,after%20surgery.",
    "https://lungfoundation.com.au/blog/lobectomies-before-after-care/",
    "https://ctsurgerypatients.org/10-common-myths-about-lung-cancer-surgery",
    "https://www.svhlunghealth.com.au/rehabilitation/after-lung-surgery",
    "https://www.sciencedirect.com/science/article/pii/S0003497522006750",
    "https://pubmed.ncbi.nlm.nih.gov/33188754/#:~:text=Results%3A%20The%20mean%2090%2Dday,cost%2C%20respectively."
]
urls1 = [
    "https://academic.oup.com/ejcts/article/20/4/694/373980",
    "https://academic.oup.com/ejcts/article/40/3/715/479897#107765527",
    "https://acsjournals.onlinelibrary.wiley.com/doi/10.1002/cncr.24625",
    "https://ascopubs.org/doi/10.1200/JCO.2011.39.5269#:~:text=Treatment%2520rates%2520decreased%2520more,lower%2520rates&text=advancing%2520age%2520than%2520with,lower%2520rates&text=for%2520all%2520stages%252C%2520such,lower%2520rates&text=older%2520patients%2520with%2520no,lower%2520rates",
    "https://atlanticendomd.com/how-much-does-thyroid-surgery-cost-with-insurance/#:~:text=Understanding%2520Different%2520Thyroid%2520Surgeries,Minimally%2520invasive&text=Associated%2520Costs%2520%253B%2520Lobectomy,Minimally%2520invasive&text=of%2520one%2520lobe%2529%252C%2520%252410%252C000,Minimally%2520invasive&text=%252420%252C000%252C%2520Dependent%2520on%2520your,Minimally%2520invasive",
    "https://cardiothoracicsurgery.biomedcentral.com/articles/10.1186/s13019-021-01445-7",
    "https://columbiasurgery.org/conditions-and-treatments/heart-transplantation",
    "https://columbiasurgery.org/conditions-and-treatments/nonsurgical-lung-treatments#:~:text=Bronchodilators%253A%2520Bronchodilators%2520work%2520by,common%2520examples.&text=and%2520airways%2520to%2520make,common%2520examples.&text=using%2520an%2520inhaler%2520either,common%2520examples.&text=conditions%2520such%2520as%2520COPD,common%2520examples.",
    "https://columbiasurgery.org/heart-transplant/lifestyle-changes-after-your-operation",
    "https://columbiasurgery.org/lung-transplant/physical-therapy-after-lung-transplantation",
    "https://copdnewstoday.com/social-clips/myths-about-lung-transplants/3/",
    "https://cost.sidecarhealth.com/s/heart-bypass-surgery-cost-in-texas",
    "https://ctsurgerypatients.org/10-common-myths-about-lung-cancer-surgery",
    "https://ctsurgerypatients.org/adult-heart-disease/mitral-valve-disease#causes-and-symptoms-",
    "https://ctsurgerypatients.org/procedures/lobectomy",
    "https://drakhilmonga.com/blog/effective-treatment-thrombectomy-and-thrombolysis/",
    "https://drkshitijdubey.com/heart-transplant-myths-and-facts/",
    "https://drvishalkhullar.com/heart-bypass-surgery-myths/",
    "https://emedicine.medscape.com/article/1893992-overview?form=fpf",
    "https://fortune.com/2017/09/14/organ-transplant-cost/",
    "https://go2.org/resources-and-support/support-groups/",
    "https://guysandstthomasspecialistcare.co.uk/treatments/mitral-valve-repair-and-replacement/",
    "https://healthcare.utah.edu/cardiovascular/treatments/mitral-valve-repair-replacement",
    "https://healthinfo.coxhealth.com/Conditions/Cancer/92,P07749",
    "https://healthy.kaiserpermanente.org/health-wellness/health-encyclopedia/he.mitral-valve-repair-surgery-what-to-expect-at-home.acl0507",
    "https://helphopelive.org/double-lung-financial-assistance/",
    "https://helphopelive.org/heart-financial-assistance/",
    "https://intermountainhealthcare.org/ckr-ext/Dcmnt?ncid=521397147",
    "https://jamanetwork.com/journals/jama/fullarticle/2816670",
    "https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2794707",
    "https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2803950",
    "https://journal.chestnet.org/article/S0096-0217(15)34296-5/fulltext",
    "https://journalofethics.ama-assn.org/article/how-should-physicians-respond-requests-lvad-removal/2019-05",
    "https://journalofethics.ama-assn.org/article/indications-bypass-surgery/2004-02",
    "https://journals.lww.com/asaiojournal/fulltext/2020/08000/cost_effectiveness_of_thoracotomy_approach_for_the.4.aspx#:~:text=LVAD%2520patients%2520had%2520higher,MM%2520%2524334%252C117.&text=costs%2520and%2520higher%2520lifetime,MM%2520%2524334%252C117.&text=Total%2520cost%2520for%2520the,MM%2520%2524334%252C117.&text=arm%2520was%2520%2524551%252C934%2520and,MM%2520%2524334%252C117.",
    # Add remaining URLs similarly...
]
urls2 = [
    "https://jtd.amegroups.org/article/view/7641/html",
    "https://link.springer.com/referenceworkentry/10.1007/978-3-030-36123-5_7",
    "https://livelyme.com/blog/the-costs-of-bypass-surgery",
    "https://lungfoundation.com.au/blog/lobectomies-before-after-care/",
    "https://lungfoundation.com.au/blog/understanding-lobectomies/",
    "https://med.uth.edu/cvs/patient-care/conditionsandprocedures/video-assisted-thoracic-surgery-vats/recovery-after-vats-and-robotic-lobectomy-surgery/",
    "https://medicalupdate.pennstatehealth.org/cardiology/heart-failure-treatment/",
    "https://medicine.umich.edu/dept/cardiac-surgery/patient-information/adult-cardiac-surgery/adult-conditions-treatments/heart-transplant",
    "https://medlineplus.gov/ency/article/001124.htm",
    "https://medlineplus.gov/ency/article/007411.htm",
    "https://mirm-pitt.net/double-lung-transplant-done-twice/",
    "https://my.clevelandclinic.org/departments/transplant/programs/heart/process",
    "https://my.clevelandclinic.org/departments/transplant/programs/heart/recovery#:~:text=At%2520first%252C%2520you%2520may,proper%2520healing%253A&text=or%2520incision%2520discomfort%2520in,proper%2520healing%253A&text=Itching%252C%2520tightness%252C%2520or%2520numbness,proper%2520healing%253A&text=also%2520normal.%2520Follow%2520the,proper%2520healing%253A",
    "https://my.clevelandclinic.org/health/body/21704-heart",
    "https://my.clevelandclinic.org/health/diseases/22242-thrombosis",
    "https://my.clevelandclinic.org/health/treatments/16897-coronary-artery-bypass-surgery",
    "https://my.clevelandclinic.org/health/treatments/16949-enhanced-external-counterpulsation-eecp",
    "https://my.clevelandclinic.org/health/treatments/17087-heart-transplant",
    "https://my.clevelandclinic.org/health/treatments/17240-mitral-valve-repair",
    "https://my.clevelandclinic.org/health/treatments/17608-lobectomy",
    "https://my.clevelandclinic.org/health/treatments/17608-lobectomy#procedure-details",
    "https://my.clevelandclinic.org/health/treatments/22897-thrombectomy",
    "https://my.clevelandclinic.org/health/treatments/23044-lung-transplant",
    "https://myhealth.alberta.ca/Health/aftercareinformation/pages/conditions.aspx?hwid=abk5351",
    "https://myhealth.alberta.ca/Health/aftercareinformation/pages/conditions.aspx?hwid=abs2466",
    "https://myhealth.alberta.ca/Health/aftercareinformation/pages/conditions.aspx?hwid=zc2605",
    "https://myhealth.alberta.ca/Health/aftercareinformation/pages/conditions.aspx?hwid=zy1364",
    "https://myhealth.alberta.ca/Health/aftercareinformation/pages/conditions.aspx?hwid=zy1364#:~:text=After%2520surgery%252C%2520you%2520will,%2528nasal%2520cannula%2529.&text=respiratory%2520therapist%2520will%2520teach,%2528nasal%2520cannula%2529.&text=get%2520as%2520much%2520oxygen,%2528nasal%2520cannula%2529.&text=get%2520extra%2520oxygen%2520through,%2528nasal%2520cannula%2529.",
    "https://myhealth.alberta.ca/Health/pages/conditions.aspx?hwid=ue4713abc",
    "https://news.weill.cornell.edu/news/2020/12/long-term-outcomes-after-coronary-artery-bypass-surgery-differ-by-sex",
    "https://newsnetwork.mayoclinic.org/discussion/beating-the-odds-for-a-transplant/",
    "https://newsnetwork.mayoclinic.org/discussion/total-hospital-cost-of-robotic-or-conventional-open-chest-mitral-valve-repair-surgery-is-similar-2/",
    "https://nyulangone.org/conditions/chronic-obstructive-pulmonary-disease/treatments/lifestyle-changes-for-chronic-obstructive-pulmonary-disease",
    "https://nyulangone.org/conditions/deep-vein-thrombosis/treatments/minimally-invasive-procedures-for-deep-vein-thrombosis",
    "https://optn.transplant.hrsa.gov/news/new-lung-allocation-policy-in-effect/",
    "https://optn.transplant.hrsa.gov/patients/by-organ/heart/questions-and-answers-for-transplant-candidates-about-the-adult-heart-allocation-system/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC1070271/#:~:text=OBJECTIVE%253A%2520To%2520examine%2520the,labeled%2520%2522HMO.%2522&text=Our%2520sample%2520consists%2520of,labeled%2520%2522HMO.%2522&text=to%2520estimate%2520the%2520probability,labeled%2520%2522HMO.%2522&text=quality%2520had%2520a%2520greater,labeled%2520%2522HMO.%2522",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC4250557/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC4314218/#:~:text=Women%2520with%2520lung%2520cancer,on%2520sex.&text=stage.%2520The%2520etiology%2520of,on%2520sex.&text=survival%2520differences%2520could%2520be,on%2520sex.&text=This%2520study%2520was%2520undertaken,on%2520sex.",
    # Add remaining URLs in similar fashion...
]

urls=urls+urls_doc
#urls1+urls2
urls=list(set(urls)) #remove duplicates
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




#RAG chain component STARTS here


vectorstore = Chroma.from_documents(documents=documents, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever().with_config(
    tags=["retriever"]
)



llm = ChatOpenAI(model="gpt-4o-mini",temperature=0,streaming=True)
llm_summary = ChatOpenAI(model="gpt-4o-mini",temperature=0,streaming=False)
#llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, streaming=True)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is. Language: {language}"""
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
Use ten sentences maximum.Language: {language}\
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
app.add_middleware(SessionMiddleware, secret_key="health123",max_age=None)

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


async def generate_chat_events(request: Request, message: dict):

    # print("===================== START Chat model has started. generate_chat_events ====================")
    # print(message)
    # print(message)
    language = message.get("language", "en")
    session =  request.session
    session_chat_history = session.get("session_chat_history", [])


    session_chat_history.append({"User Question":message})
    # print("=====================END Chat model has started. generate_chat_events ====================")
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
                # print("Chat model has completed one response.")
                if final_message != "":
                    print("====================================FINAL output==============================")
                    print("Final message:", final_message)
                    session_chat_history.append({"AI Response": final_message})
                    session["session_chat_history"] = session_chat_history
                    # print("====================================END FINAL output==============================")

            session["session_chat_history"] = session_chat_history
    except Exception as e:
        print('error' + str(e))
        #session["session_chat_history"] = session_chat_history


from io import BytesIO








@app.post("/export_pdf")
async def export_chat(request: Request):
    body = await request.body()
    data = json.loads(body.decode('utf-8'))
    chat_history = data.get('chat_history', [])
    language_in = data.get('language', 'en')  # Default to 'en' if not provided
    if language_in == 'en':
        language = "English"
    elif language_in == 'hi':
        language = "Hindi"
    else:
        language = "Spanish"
    template_messages = [("system", f"Format and summarize the following conversation without removing essential parts so that the human patient has all the answers. The content of the text would be entering into a pdf. Explain with proper section. Translate the summary to : {language}")]

    for chat_item in chat_history:
        if "Human" in chat_item:
            template_messages.append(("human", chat_item["Human"]))
        if "AI" in chat_item:
            template_messages.append(("ai", chat_item["AI"]))

    summary_prompt = ChatPromptTemplate.from_messages(template_messages)
    formatted_prompt = summary_prompt.format()
    summary = llm_summary.invoke(formatted_prompt)

    pdf_title = "Chat Summary"
    pdf_content = f"{pdf_title}\n\n{summary.content}"

    # Create a PDF buffer
    pdf_buffer = BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)

    # Define styles with a larger font size
    styles = getSampleStyleSheet()
    normal_style = styles['Normal']
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=12,
        fontName='NotoSansDevanagari' if language_in == 'hi' else 'Helvetica'
    )
    section_style = ParagraphStyle(
        'Section',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=8,
        fontName='NotoSansDevanagari' if language_in == 'hi' else 'Helvetica'
    )
    content_style = ParagraphStyle(
        'Content',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=6,
        fontName='NotoSansDevanagari' if language_in == 'hi' else 'Helvetica'
    )

    # Create a list to hold the PDF elements
    story = []

    # Add the title
    story.append(Paragraph(pdf_title, title_style))
    story.append(Spacer(1, 12))

    # Split the content into sections
    sections = pdf_content.split('##')
    for section in sections:
        if section.strip():
            lines = section.split('\n')
            if lines:
                # Add section title
                story.append(Paragraph(lines[0].strip(), section_style))
                story.append(Spacer(1, 8))
                # Add section content
                for line in lines[1:]:
                    if line.strip():
                        # Highlight key texts in red and remove "**"
                        line = re.sub(r'\*\*(.*?)\*\*', r'<font color="red">\1</font>', line)
                        story.append(Paragraph(line.strip(), content_style))
                        story.append(Spacer(1, 6))

    # Build the PDF
    doc.build(story)

    pdf_buffer.seek(0)


    return StreamingResponse(pdf_buffer, media_type='application/pdf', headers={'Content-Disposition': 'attachment; filename="chat_summary.pdf"'})



@app.get("/chat_stream/{message}")
async def chat_stream_events(request: Request, message: str):


    language_in = request.query_params.get('language', 'en')  # Default to 'en' if not provided
    if language_in == 'en':
        language = "English"
    elif language_in == 'hi':
        language = "Hindi"
    else:
        language = "Spanish"
    return StreamingResponse(generate_chat_events(
        request=request, message={"question": message,
                                  "chat_history": [],
                                  "language": language}),
                             media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)