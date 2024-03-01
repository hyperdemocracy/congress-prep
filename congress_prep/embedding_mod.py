import json
from pathlib import Path
import tempfile
from typing import Optional, Union

from datasets import load_dataset
from langchain_core.documents import Document
from huggingface_hub import HfApi
import numpy as np
import pandas as pd
import rich
from sentence_transformers import SentenceTransformer
import yaml

from congress_prep import utils


VEC_DTYPE = "float32"


def get_readme_str(
    model_name: str, chunk_size: int, chunk_overlap: int, congress_nums: list[int]
):

    model_tag = model_name.replace("/", "-")
    yaml_dict = {
        "configs": [
            {
                "config_name": "default",
                "data_files": [
                    {
                        "split": str(cn),
                        "path": f"data/usc-{cn}-vecs-v1-s{chunk_size}-o{chunk_overlap}-{model_tag}.parquet",
                    }
                    for cn in congress_nums
                ],
            }
        ],
        "dataset_info": {
            "features": [
                {"name": "chunk_id", "dtype": "string"},
                {"name": "text_id", "dtype": "string"},
                {"name": "legis_id", "dtype": "string"},
                {"name": "text", "dtype": "string"},
                {"name": "vec", "list": {"dtype": VEC_DTYPE}},
                {
                    "name": "metadata",
                    "struct": [
                        {"name": "chunk_id", "dtype": "string"},
                        {"name": "chunk_index", "dtype": "int32"},
                        {"name": "congress_num", "dtype": "int32"},
                        {"name": "legis_class", "dtype": "string"},
                        {"name": "legis_id", "dtype": "string"},
                        {"name": "legis_num", "dtype": "int32"},
                        {"name": "legis_type", "dtype": "string"},
                        {"name": "legis_version", "dtype": "string"},
                        {"name": "start_index", "dtype": "int32"},
                        {"name": "text_date", "dtype": "string"},
                        {"name": "text_id", "dtype": "string"},
                    ],
                },
            ]
        },
    }
    readme_str = "---\n{}---".format(yaml.safe_dump(yaml_dict))
    return readme_str


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

    chunk_tag = f"chunks-v1-s{chunk_size}-o{chunk_overlap}"
    c_tag = f"usc-{congress_num}-{chunk_tag}"
    c_fpath = congress_hf_path / f"usc-{chunk_tag}" / f"{c_tag}.parquet"
    df_c = pd.read_parquet(c_fpath)

    if nlim is not None:
        df_c = df_c.head(nlim)

    model = SentenceTransformer(model_name)
    vecs = model.encode(
        df_c["text"].tolist(),
        show_progress_bar=True,
    )

    df_c["vec"] = [row for row in vecs]
    df_c["chunk_id"] = df_c["metadata"].apply(lambda x: x["chunk_id"])
    df_c["text_id"] = df_c["metadata"].apply(lambda x: x["text_id"])
    df_c["legis_id"] = df_c["metadata"].apply(lambda x: x["legis_id"])
    col_order = ["chunk_id", "text_id", "legis_id", "text", "metadata", "vec"]
    df_c = df_c[col_order]

    out_dir = f"usc-vecs-v1-s{chunk_size}-o{chunk_overlap}-{model_tag}"
    out_path = congress_hf_path / out_dir / "data"
    out_path.mkdir(parents=True, exist_ok=True)

    v_tag = f"usc-{congress_num}-vecs-v1-s{chunk_size}-o{chunk_overlap}-{model_tag}"
    v_fpath = out_path / f"{v_tag}.parquet"
    df_c.to_parquet(v_fpath)


def upload_hf(
    congress_hf_path: Union[str, Path],
    chunk_size: int,
    chunk_overlap: int,
    model_name: str,
):

    congress_hf_path = Path(congress_hf_path)
    model_tag = model_name.replace("/", "-")

    rich.print("EMBEDDING (upload hf)")
    rich.print(f"{congress_hf_path=}")
    rich.print(f"{chunk_size=}")
    rich.print(f"{chunk_overlap=}")
    rich.print(f"{model_name=}")
    rich.print(f"{model_tag=}")

    tag = f"usc-vecs-v1-s{chunk_size}-o{chunk_overlap}-{model_tag}"
    upload_folder = congress_hf_path / tag
    repo_id = f"hyperdemocracy/{tag}"
    rich.print(f"{repo_id=}")
    rich.print(f"{upload_folder=}")

    api = HfApi()
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=upload_folder,
#        path_in_repo="",
        repo_id=repo_id,
        repo_type="dataset",
    )


def write_readme(
    congress_hf_path: Union[Path, str],
    model_name: str,
    chunk_size: int,
    chunk_overlap: int,
    congress_nums: list[int],
):

    model_tag = model_name.replace("/", "-")
    out_dir = f"usc-vecs-v1-s{chunk_size}-o{chunk_overlap}-{model_tag}"
    out_path = Path(congress_hf_path) / out_dir
    out_path.mkdir(parents=True, exist_ok=True)
    fpath = Path(out_path) / "README.md"
    readme_str = get_readme_str(model_name, chunk_size, chunk_overlap, congress_nums)
    with open(fpath, "w") as fp:
        fp.write(readme_str)


if __name__ == "__main__":

    congress_hf_path = Path("/Users/galtay/data/congress-hf")
    congress_nums = [113, 114, 115, 116, 117, 118]
    chunk_size = 1024
    chunk_overlap = 256
    nlim = None
    model_names = ["BAAI/bge-small-en-v1.5", "BAAI/bge-large-en-v1.5"]

    do_write_local = False
    do_upload_hf = False

    for model_name in model_names:
        if do_write_local:
            write_readme(congress_hf_path, model_name, chunk_size, chunk_overlap, congress_nums)
            for congress_num in congress_nums:
                write_local(
                    congress_hf_path,
                    congress_num,
                    chunk_size,
                    chunk_overlap,
                    model_name,
                    nlim=nlim,
                )

        if do_upload_hf:
            upload_hf(
                congress_hf_path,
                chunk_size,
                chunk_overlap,
                model_name,
            )
