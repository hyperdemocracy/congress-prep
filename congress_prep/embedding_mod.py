import json
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from langchain_core.documents import Document
from huggingface_hub import HfApi
import pandas as pd
import rich
from sentence_transformers import SentenceTransformer


def embed_dataset(
    cn: int,
    chunk_size: int,
    chunk_overlap: int,
    model_name: str,
    model_tag: str,
    upload: bool = False,
    nlim: Optional[int] = None,
):

    congress_hf_path = Path("/Users/galtay/data/congress-hf")
    f_in = congress_hf_path / f"usc-{cn}-chunks-v1-s{chunk_size}-o{chunk_overlap}.parquet"
    df_in = pd.read_parquet(f_in)
    if nlim is not None:
        df_in = df_in.head(nlim)


    model = SentenceTransformer(model_name)
    vecs = model.encode(
        df_in["page_content"].tolist(),
        show_progress_bar=True,
    )
    df_text = pd.DataFrame({"text": df_in["page_content"].tolist()})
    df_vec = pd.DataFrame({"vec": vecs.tolist()})
    df = pd.concat([
        df_text,
        df_in[["metadata"]],
        df_vec,
    ], axis=1)
    df["legis_id"] = df["metadata"].apply(lambda x: x["legis_id"])
    df["text_id"] = df["metadata"].apply(lambda x: x["text_id"])
    df["chunk_id"] = df["metadata"].apply(lambda x: x["chunk_id"])
    col_order = ["chunk_id", "text_id", "legis_id", "text", "metadata", "vec"]
    df = df[col_order]

    repo_tag = f"usc-{cn}-vecs-v1-s{chunk_size}-o{chunk_overlap}-{model_tag}"
    fout = congress_hf_path / f"{repo_tag}.parquet"
    df.to_parquet(fout)

    if upload:
        rich.print(f"{repo_tag=}")
        api = HfApi()
        repo_id = f"hyperdemocracy/{repo_tag}"
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            exist_ok=True,
        )
        api.upload_file(
            path_or_fileobj=fout,
            path_in_repo=fout.name,
            repo_id=repo_id,
            repo_type="dataset",
        )


if __name__ == "__main__":
    cns = [113, 114, 115, 116, 117, 118]
    chunk_size = 1024
    chunk_overlap = 256
    model_name = "BAAI/bge-small-en-v1.5"
    model_tag = "bge-small-en-v1p5"
    upload = False
    nlim = None

    for cn in cns:
        embed_dataset(
            cn,
            chunk_size,
            chunk_overlap,
            model_name,
            model_tag,
            upload=upload,
            nlim=nlim,
        )
        break
