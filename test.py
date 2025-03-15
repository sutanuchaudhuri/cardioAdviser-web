
from bs4 import BeautifulSoup
from langchain_core.documents import Document


html_docs={"a.html":"https://www.pennmedicine.org/make-an-appointment"
           ,"b.html":"https://www.pennmedicine.org/providers?keyword=Penn-Heart-Surgery-Program&keywordid=57534&keywordtypeid=11"
           ,"c.html":"https://www.hopkinsmedicine.org/heart-vascular-institute/cardiac-surgery"
           # ,"d.html":"https://www.hopkinsmedicine.org/heart-vascular-institute/cardiac-surgery/mitral-valve-repair-replacement"
           }


def extract_html_text(html_content_file_name,reference_url):
    soup = BeautifulSoup(open(f"resources/{html_content_file_name}", encoding="utf8"), "html.parser")
    html_text= soup.get_text(separator=" ", strip=True)
    document=Document(page_content=html_text, metadata={'source': f'{reference_url}'})
    return document



documents=[]
for html_doc_key in html_docs:
   documents.append(extract_html_text(html_content_file_name=html_doc_key,reference_url=html_docs[html_doc_key]))
print(documents[1])




#print(list(html_docs.keys()))