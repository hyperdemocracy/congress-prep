"""
metadata={'chunk_id': '114-s-2604-is-2', 'chunk_index': 2, 'congress_num': 114, 'introduced_date': '2016-02-29', 'legis_class': 'bills', 'legis_id': '114-s-2604', 'legis_num': 2604, 'legis_type': 's', 'legis_version': 'is', 'origin_chamber': 'Senate', 'sponsor': 'W000805', 'start_index': 1652, 'text_date': '2016-02-29T05:00:00Z', 'text_id': '114-s-2604-is', 'update_date': '2023-01-11T13:31:58Z'}
"""

from pathlib import Path
from langchain_openai import ChatOpenAI
from congress_prep.chroma_mod import load_langchain_db
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


congress_hf_path = Path("/Users/galtay/data/congress-hf")
cns = (113, 114, 115, 116, 117, 118)
chunk_size = 1024
chunk_overlap = 256
model_name = "BAAI/bge-small-en-v1.5"


vectorstore = load_langchain_db(
    congress_hf_path,
    cns,
    chunk_size,
    chunk_overlap,
    model_name,
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 16},
)


#model_name = "gpt-4-0125-preview"
model_name = "gpt-3.5-turbo-0125"
model = ChatOpenAI(name=model_name, temperature=0)

template = """Answer the question based only on the following snippets from congressional legislation.
Reference the legislation ID for snippets that are useful in answering the question.

{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)


def format_doc(idoc, doc):
    return f"Snippet {idoc}\nLegislation ID: {doc.metadata['legis_id']}\n ... {doc.page_content} ..."


def format_docs(docs):
    return "\n\n".join([format_doc(idoc, doc) for idoc, doc in enumerate(docs)])

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
