import json
import os
import tiktoken
import pandas as pd
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings

load_dotenv()

VECTOR_DB_FOLDER = "Vector-DB"
os.makedirs(VECTOR_DB_FOLDER, exist_ok=True)

def readFile(filepath):
    data = pd.read_csv(filepath)
    return data

def create_docs(data):

    docs = []
    total_tokens = 0
    
    encoding = tiktoken.encoding_for_model('text-embedding-3-large')

    for index, row in data.iterrows():
        doc = {
            "resume": row["Resume_str"],
            "category": row["Category"],
            "source": f'{row["ID"]}.pdf',
        }
        doc = json.dumps(doc)
        metadata={"source" : f'{row["ID"]}.pdf'}
        num_tokens = len(encoding.encode(doc))
        total_tokens += num_tokens
        docs.append(Document(page_content=doc, metadata=metadata))
    
    print(f'Total Tokens: {total_tokens}')
    return docs


def create_vector_db(docs, PersistDir):

    embedding_model = OpenAIEmbeddings(model='text-embedding-3-large',show_progress_bar=True)
    vectordb = Chroma.from_documents(
            documents=docs,
            embedding=embedding_model,
            persist_directory=PersistDir,
    )
    vectordb.persist()
    
data = readFile('./Dataset/Resume.csv')
docs = create_docs(data)
create_vector_db(docs, VECTOR_DB_FOLDER)


