from pathlib import Path
from congress_prep.chroma_mod import load_collection
from congress_prep.chroma_mod import load_langchain_db


congress_hf_path = Path("/Users/galtay/data/congress-hf")
cns = (113, 114, 115, 116, 117, 118)
chunk_size = 1024
chunk_overlap = 256
model_name = "BAAI/bge-small-en-v1.5"


clients = {}
collections = {}
langchain_dbs = {}


clients[cns], collections[cns] = load_collection(
    congress_hf_path,
    cns,
    chunk_size,
    chunk_overlap,
    model_name,
)
langchain_dbs[cns] = load_langchain_db(
    congress_hf_path,
    cns,
    chunk_size,
    chunk_overlap,
    model_name,
)

for cn in cns:

    clients[cn], collections[cn] = load_collection(
        congress_hf_path,
        [cn],
        chunk_size,
        chunk_overlap,
        model_name,
    )
    langchain_dbs[cn] = load_langchain_db(
        congress_hf_path,
        [cn],
        chunk_size,
        chunk_overlap,
        model_name,
    )
