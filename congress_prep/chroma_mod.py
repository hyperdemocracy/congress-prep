import json
from pathlib import Path
from typing import Optional, Union
import shutil

import chromadb
from chromadb.utils import embedding_functions
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
import numpy as np
import pandas as pd
import rich
from sklearn.preprocessing import normalize
from tqdm import tqdm


def create_index(
    congress_hf_path: Union[str, Path],
    congress_nums: list[int],
    chunk_size: int,
    chunk_overlap: int,
    model_name: str,
    batch_size: int = 1000,
    nlim: Optional[int] = None,
):

    congress_hf_path = Path(congress_hf_path)
    model_tag = model_name.replace("/", "-")
    cn_tag = f"{congress_nums[0]}-to-{congress_nums[-1]}"

    rich.print(f"{congress_hf_path=}")
    rich.print(f"{congress_nums=}")
    rich.print(f"{chunk_size=}")
    rich.print(f"{chunk_overlap=}")
    rich.print(f"{model_name=}")
    rich.print(f"{model_tag=}")
    rich.print(f"{cn_tag=}")

    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    emb_fn = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        query_instruction="Represent this question for searching relevant passages: ",
    )

    # tag for all congress nums
    persistent_tag = f"usc-{cn_tag}-vecs-v1-s{chunk_size}-o{chunk_overlap}-{model_tag}"
    persistent_path = congress_hf_path / f"{persistent_tag}-chromadb"
    rich.print(f"{persistent_path=}")
    if persistent_path.exists():
        shutil.rmtree(persistent_path)

    chroma_client = chromadb.PersistentClient(path=str(persistent_path))
    collection = chroma_client.create_collection(
        name=persistent_tag, get_or_create=True
    )

    for cn in congress_nums:

        tag = f"usc-{cn}-vecs-v1-s{chunk_size}-o{chunk_overlap}-{model_tag}"
        rich.print(f"{tag=}")
        df = pd.read_parquet(congress_hf_path / f"{tag}.parquet")
        if nlim is not None:
            df = df.head(nlim)

        assert df["chunk_id"].nunique() == df.shape[0]
        ii_los = list(range(0, df.shape[0], batch_size))
        for ii_lo in tqdm(ii_los):

            df_batch = df.iloc[ii_lo : ii_lo + batch_size]
            vecs = np.stack(df_batch["vec"].values)
            vecs = normalize(vecs)

            metas = df_batch["metadata"].tolist()
            metas = [
                {k: str(v) if v is None else v for k, v in meta.items()}
                for meta in metas
            ]

            collection.add(
                embeddings=vecs.tolist(),
                documents=df_batch["text"].tolist(),
                metadatas=metas,
                ids=df_batch["chunk_id"].to_list(),
            )


def load_collection(
    congress_hf_path: Union[str, Path],
    congress_nums: list[int],
    chunk_size: int,
    chunk_overlap: int,
    model_name: str,
):

    congress_hf_path = Path(congress_hf_path)
    model_tag = model_name.replace("/", "-")
    cn_tag = f"{congress_nums[0]}-to-{congress_nums[-1]}"
    persistent_tag = f"usc-{cn_tag}-vecs-v1-s{chunk_size}-o{chunk_overlap}-{model_tag}"
    persistent_path = congress_hf_path / f"{persistent_tag}-chromadb"

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name, normalize_embeddings=True
    )
    chroma_client = chromadb.PersistentClient(path=str(persistent_path))
    collection = chroma_client.get_collection(
        name=persistent_tag, embedding_function=ef
    )
    return chroma_client, collection


def load_langchain_db(
    congress_hf_path: Union[str, Path],
    congress_nums: list[int],
    chunk_size: int,
    chunk_overlap: int,
    model_name: str,
):

    congress_hf_path = Path(congress_hf_path)
    model_tag = model_name.replace("/", "-")
    cn_tag = f"{congress_nums[0]}-to-{congress_nums[-1]}"
    persistent_tag = f"usc-{cn_tag}-vecs-v1-s{chunk_size}-o{chunk_overlap}-{model_tag}"
    persistent_path = congress_hf_path / f"{persistent_tag}-chromadb"

    chroma_client = chromadb.PersistentClient(path=str(persistent_path))
    collection = chroma_client.get_collection(name=persistent_tag)

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
        collection_name=persistent_tag,
        embedding_function=emb_fn,
    )
    return langchain_db


if __name__ == "__main__":
    congress_hf_path = Path("/Users/galtay/data/congress-hf")
    cns = [113, 114, 115, 116, 117, 118]
    chunk_size = 1024
    chunk_overlap = 256
    model_name = "BAAI/bge-small-en-v1.5"
    batch_size = 5000
    nlim = None

    # create one chroma index with all congress nums
    create_index(
        congress_hf_path,
        cns,
        chunk_size,
        chunk_overlap,
        model_name,
        batch_size=batch_size,
        nlim=nlim,
    )

    # and create one for each congress num
    for cn in cns:

        create_index(
            congress_hf_path,
            [cn],
            chunk_size,
            chunk_overlap,
            model_name,
            batch_size=batch_size,
            nlim=nlim,
        )
