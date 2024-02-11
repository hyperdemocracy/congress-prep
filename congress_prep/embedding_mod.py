import json
from pathlib import Path
from typing import Optional, Union

from datasets import load_dataset
from langchain_core.documents import Document
from huggingface_hub import HfApi
import pandas as pd
import rich
from sentence_transformers import SentenceTransformer


def write_local(
    congress_hf_path: Union[str, Path],
    congress_num: int,
    chunk_size: int,
    chunk_overlap: int,
    model_name: str,
    nlim: Optional[int] = None,
):

    congress_hf_path = Path(congress_hf_path)
    model_tag = model_name.replace("/", "-")

    rich.print(f"{congress_hf_path=}")
    rich.print(f"{congress_num=}")
    rich.print(f"{chunk_size=}")
    rich.print(f"{chunk_overlap=}")
    rich.print(f"{model_name=}")
    rich.print(f"{model_tag=}")

    tag = f"usc-{congress_num}-chunks-v1-s{chunk_size}-o{chunk_overlap}"
    f_in = congress_hf_path / f"{tag}.parquet"
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
    df = pd.concat(
        [
            df_text,
            df_in[["metadata"]],
            df_vec,
        ],
        axis=1,
    )
    df["legis_id"] = df["metadata"].apply(lambda x: x["legis_id"])
    df["text_id"] = df["metadata"].apply(lambda x: x["text_id"])
    df["chunk_id"] = df["metadata"].apply(lambda x: x["chunk_id"])
    col_order = ["chunk_id", "text_id", "legis_id", "text", "metadata", "vec"]
    df = df[col_order]

    tag = f"usc-{congress_num}-vecs-v1-s{chunk_size}-o{chunk_overlap}-{model_tag}"
    fout = congress_hf_path / f"{tag}.parquet"
    df.to_parquet(fout)


def upload_hf(
    congress_hf_path: Union[str, Path],
    congress_num: int,
    chunk_size: int,
    chunk_overlap: int,
    model_name: str,
):

    congress_hf_path = Path(congress_hf_path)
    model_tag = model_name.replace("/", "-")

    rich.print(f"{congress_hf_path=}")
    rich.print(f"{congress_num=}")
    rich.print(f"{chunk_size=}")
    rich.print(f"{chunk_overlap=}")
    rich.print(f"{model_name=}")
    rich.print(f"{model_tag=}")

    tag = f"usc-{congress_num}-vecs-v1-s{chunk_size}-o{chunk_overlap}-{model_tag}"
    fpath = congress_hf_path / f"{tag}.parquet"

    rich.print(f"{tag=}")
    api = HfApi()
    repo_id = f"hyperdemocracy/{tag}"
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        exist_ok=True,
    )
    api.upload_file(
        path_or_fileobj=fpath,
        path_in_repo=fpath.name,
        repo_id=repo_id,
        repo_type="dataset",
    )


if __name__ == "__main__":
    congress_hf_path = Path("/Users/galtay/data/congress-hf")
    congress_nums = [113, 114, 115, 116, 117, 118]
    chunk_size = 1024
    chunk_overlap = 256
    nlim = None
    model_names = ["BAAI/bge-small-en-v1.5", "BAAI/bge-large-en-v1.5"]
    for model_name in model_names[1:]:
        for congress_num in congress_nums:
            #            write_local(
            #                congress_hf_path,
            #                congress_num,
            #                chunk_size,
            #                chunk_overlap,
            #                model_name,
            #                nlim=nlim,
            #            )
            upload_hf(
                congress_hf_path,
                congress_num,
                chunk_size,
                chunk_overlap,
                model_name,
            )
