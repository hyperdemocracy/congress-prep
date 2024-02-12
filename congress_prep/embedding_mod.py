import json
from pathlib import Path
import tempfile
from typing import Optional, Union

from datasets import load_dataset
from langchain_core.documents import Document
from huggingface_hub import HfApi
import pandas as pd
import rich
from sentence_transformers import SentenceTransformer

from congress_prep import utils


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

    rich.print("EMBEDDING (write local)")
    rich.print(f"{congress_hf_path=}")
    rich.print(f"{congress_num=}")
    rich.print(f"{chunk_size=}")
    rich.print(f"{chunk_overlap=}")
    rich.print(f"{model_name=}")
    rich.print(f"{model_tag=}")

    c_tag = f"usc-{congress_num}-chunks-v1-s{chunk_size}-o{chunk_overlap}"
    c_fpath = congress_hf_path / f"{c_tag}.parquet"
    df_c = pd.read_parquet(c_fpath)
    if nlim is not None:
        df_c = df_c.head(nlim)

    model = SentenceTransformer(model_name)
    vecs = model.encode(
        df_c["text"].tolist(),
        show_progress_bar=True,
    )
    df_text = pd.DataFrame({"text": df_c["text"].tolist()})
    df_vec = pd.DataFrame({"vec": vecs.tolist()})
    df_v = pd.concat(
        [
            df_text,
            df_c[["metadata"]],
            df_vec,
        ],
        axis=1,
    )
    df_v["chunk_id"] = df_v["metadata"].apply(lambda x: x["chunk_id"])
    df_v["text_id"] = df_v["metadata"].apply(lambda x: x["text_id"])
    df_v["legis_id"] = df_v["metadata"].apply(lambda x: x["legis_id"])
    col_order = ["chunk_id", "text_id", "legis_id", "text", "metadata", "vec"]
    df_v = df_v[col_order]

    v_tag = f"usc-{congress_num}-vecs-v1-s{chunk_size}-o{chunk_overlap}-{model_tag}"
    v_fpath = congress_hf_path / f"{v_tag}.parquet"
    df_v.to_parquet(v_fpath)


def upload_hf(
    congress_hf_path: Union[str, Path],
    congress_num: int,
    chunk_size: int,
    chunk_overlap: int,
    model_name: str,
):

    congress_hf_path = Path(congress_hf_path)
    model_tag = model_name.replace("/", "-")

    rich.print("EMBEDDING (upload hf)")
    rich.print(f"{congress_hf_path=}")
    rich.print(f"{congress_num=}")
    rich.print(f"{chunk_size=}")
    rich.print(f"{chunk_overlap=}")
    rich.print(f"{model_name=}")
    rich.print(f"{model_tag=}")

    v_tag = f"usc-{congress_num}-vecs-v1-s{chunk_size}-o{chunk_overlap}-{model_tag}"
    v_fpath = congress_hf_path / f"{v_tag}.parquet"
    rich.print(f"{v_fpath=}")
    if not v_fpath.exists():
        rich.print(f"{v_fpath} does not exist")
        return
    df_v = pd.read_parquet(v_fpath)
    df_v["metadata"] = df_v["metadata"].apply(
        lambda x: {k: v for k, v in x.items() if k != "type"}
    )

    c_tag = f"usc-{congress_num}-chunks-v1-s{chunk_size}-o{chunk_overlap}"
    c_fpath = congress_hf_path / f"{c_tag}.parquet"
    rich.print(f"{c_fpath=}")
    if not c_fpath.exists():
        rich.print(f"{c_fpath} does not exist")
        return
    df_c = pd.read_parquet(c_fpath)

    # merge metadata from chunking and embedding
    df_out = pd.merge(
        df_v,
        df_c[["chunk_id", "metadata"]],
        on="chunk_id",
    )
    df_out["metadata"] = df_out.apply(
        lambda z: {**z["metadata_x"], **z["metadata_y"]}, axis=1
    )
    df_out = df_out.drop(columns=["metadata_x", "metadata_y"])
    cols = ["chunk_id", "text_id", "legis_id", "text", "metadata", "vec"]
    df_out = df_out[cols]

    # overwrite original
    df_out.to_parquet(v_fpath)

    api = HfApi()
    repo_id = f"hyperdemocracy/{v_tag}"
    rich.print(f"{repo_id=}")
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        exist_ok=True,
    )
    api.upload_file(
        path_or_fileobj=v_fpath,
        path_in_repo=v_fpath.name,
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
