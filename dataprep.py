import os
import time
from dotenv import load_dotenv
from langchain_text_splitters import LatexTextSplitter
from pinecone import Pinecone
from collections import Counter
from pprint import pprint
load_dotenv("lc/api.env")
api_keyp = os.getenv("PINECONE_KEY")
pc = Pinecone(api_key=api_keyp)


def remove_repeated_lines(text, threshold=2):
    lines = text.split('\n')
    normalized_lines = [line.strip() for line in lines if line.strip()]
    line_counts = Counter(normalized_lines)
    filtered_lines = [
        line for line in normalized_lines
        if line_counts[line] <= threshold
    ]
    return " ".join(filtered_lines)

def text_split_generator(file_path:str):
    file = open(file_path,"r")
    text = file.read()
    ntext = remove_repeated_lines(text)
    latex_splliter = LatexTextSplitter(chunk_size=128,chunk_overlap=20)
    docs = latex_splliter.create_documents([ntext])
    file.close()
    file_name = file_path.split("/")[-1]
    return(docs,file_name)

def file_checker(file_name):
    index_name = "sparse-index"
    index = pc.Index(index_name)
    search = index.search_records(
    namespace="test-namespace", 
    query={
        "inputs": {"text": file_name}, 
        "top_k": 3
    },
    fields=["chunk_text"]
    )   

    if search and search.get('result', {}).get('hits'):
        for hit in search['result']['hits']:
            chunk = hit.get('fields', {}).get('chunk_text', '')
            if chunk == file_name:
                return (True,hit['_id'])
    return False

def split_list(input_list, chunk_size=96):
    if len(input_list)>96:
        return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]
    else:
        return [input_list]

def rec_generator(record:list,last_index):
    return list(map(lambda i: {"_id": f"{i}", "chunk_text": f"{record[i-last_index-1]}"}, range(last_index+1,last_index+len(record)+1)))

def get_last_index(index_name:str,namespace_name:str):
    cursor = pc.Index(index_name)
    records = list(cursor.list(namespace = namespace_name))  
    a = []
    for item in records:
        items = list(map(lambda x : int(x), item))
        a.append(items)
        b = []
        for i in a:
            b.append(max(i))
    return(max(b))



def upload(record_list:list,file_name:str):
    scursor = pc.Index("sparse-index")
    num = get_last_index("sparse-index","test-namespace")
    record =[{
        "_id" : str(num+1),
        "chunk_text" : file_name
    }]
    scursor.upsert_records("test-namespace",record)
    time.sleep(2)
    dcursor = pc.Index("dense-index")
    for item in record_list:
        record = rec_generator(item,get_last_index("dense-index","sample-namespace"))
        dcursor.upsert_records("sample-namespace",record)
        time.sleep(5)
    print("Upload Complete for file ",file_name)

def main(file_path):
    file_name = file_path.split("/")[-1]
    a,b = file_checker(file_name)
    if not a:
        chunk,file_name = text_split_generator(file_path)
        chunk_list = split_list(chunk)
        upload(chunk_list,file_name)
    else:
        print("Present at ID, ",b)

