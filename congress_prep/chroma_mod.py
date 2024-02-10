import json
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
import numpy as np
import pandas as pd
import rich
from sklearn.preprocessing import normalize


def create_index(
    cn: int,
    chunk_size: int,
    chunk_overlap: int,
    model_name: str,
    model_tag: str,
    batch_size: int = 1000,
    nlim: Optional[int] = None,
):

    congress_hf_path = Path("/Users/galtay/data/congress-hf")
    repo_tag = f"usc-{cn}-vecs-v1-s{chunk_size}-o{chunk_overlap}-{model_tag}"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    emb_fn = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        query_instruction="Represent this question for searching relevant passages: ",
    )

    chroma_client = chromadb.PersistentClient(
        path=str(congress_hf_path / f"{repo_tag}-chromadb")
    )
    try:
        chroma_client.delete_collection(name=repo_tag)
    except ValueError:
        print("collection does not exist")
    collection = chroma_client.create_collection(name=repo_tag, get_or_create=True)

    fin = congress_hf_path / f"{repo_tag}.parquet"
    df = pd.read_parquet(fin)
    if nlim is not None:
        df = df.head(nlim)

    assert df["chunk_id"].nunique() == df.shape[0]
    n_batches = max(1, df.shape[0] // batch_size)
    for df_batch in np.array_split(df, n_batches):

        vecs = np.stack(df_batch["vec"].values)
        vecs = normalize(vecs)

        metas = df_batch["metadata"].tolist()
        metas = [
            {k: str(v) if v is None else v for k, v in meta.items()} for meta in metas
        ]

        collection.add(
            embeddings=vecs.tolist(),
            documents=df_batch["text"].tolist(),
            metadatas=metas,
            ids=df_batch["chunk_id"].to_list(),
        )


def load_collection(
    cn: int,
    chunk_size: int,
    chunk_overlap: int,
    model_name: str,
    model_tag: str,
):

    congress_hf_path = Path("/Users/galtay/data/congress-hf")
    repo_tag = f"usc-{cn}-vecs-v1-s{chunk_size}-o{chunk_overlap}-{model_tag}"
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name, normalize_embeddings=True
    )
    chroma_client = chromadb.PersistentClient(
        path=str(congress_hf_path / f"{repo_tag}-chromadb")
    )
    collection = chroma_client.get_collection(name=repo_tag, embedding_function=ef)
    return chroma_client, collection


def load_langchain_db(
    cn: int,
    chunk_size: int,
    chunk_overlap: int,
    model_name: str,
    model_tag: str,
):

    congress_hf_path = Path("/Users/galtay/data/congress-hf")
    repo_tag = f"usc-{cn}-vecs-v1-s{chunk_size}-o{chunk_overlap}-{model_tag}"
    chroma_client = chromadb.PersistentClient(
        path=str(congress_hf_path / f"{repo_tag}-chromadb")
    )
    collection = chroma_client.get_collection(name=repo_tag)

    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    emb_fn = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        query_instruction="Represent this question for searching relevant passages: ",
    )
    langchain_db = Chroma(
        client=chroma_client,
        collection_name=repo_tag,
        embedding_function=emb_fn,
    )
    return langchain_db


if __name__ == "__main__":
    cns = [113, 114, 115, 116, 117, 118]
    chunk_size = 1024
    chunk_overlap = 256
    model_name = "BAAI/bge-small-en-v1.5"
    model_tag = "bge-small-en-v1p5"
    batch_size = 1000
    nlim = None

    for cn in cns:

        client, collection = load_collection(
            cn,
            chunk_size,
            chunk_overlap,
            model_name,
            model_tag,
        )
        db = load_langchain_db(
            cn,
            chunk_size,
            chunk_overlap,
            model_name,
            model_tag,
        )
        break

        create_index(
            cn,
            chunk_size,
            chunk_overlap,
            model_name,
            model_tag,
            batch_size=batch_size,
            nlim=nlim,
        )

        break
