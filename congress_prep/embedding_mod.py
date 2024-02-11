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


def metadata_from_unified_row(urow: pd.Series):
    if len(urow["text_versions"]) == 0:
        return {}
    else:
        return {
            "text_id": urow["text_versions"][0]["text_id"],
            "legis_version": urow["text_versions"][0]["legis_version"],
            "legis_class": urow["text_versions"][0]["legis_class"],
            "legis_id": urow["legis_id"],
            "congress_num": urow["congress_num"],
            "legis_type": urow["legis_type"],
            "legis_num": urow["legis_num"],
            "origin_chamber": urow["origin_chamber"],
            "update_date": urow["update_date"],
            "text_date": urow["text_versions"][0]["bs_date"],
            "introduced_date": urow["introduced_date"],
            "sponsor": urow["sponsors"][0]["bioguide_id"],
        }


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
        df_in["text"].tolist(),
        show_progress_bar=True,
    )
    df_text = pd.DataFrame({"text": df_in["text"].tolist()})
    df_vec = pd.DataFrame({"vec": vecs.tolist()})
    df = pd.concat(
        [
            df_text,
            df_in[["metadata"]],
            df_vec,
        ],
        axis=1,
    )
    df["chunk_id"] = df["metadata"].apply(lambda x: x["chunk_id"])
    df["text_id"] = df["metadata"].apply(lambda x: x["text_id"])
    df["legis_id"] = df["metadata"].apply(lambda x: x["legis_id"])
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

    v_tag = f"usc-{congress_num}-vecs-v1-s{chunk_size}-o{chunk_overlap}-{model_tag}"
    v_fpath = congress_hf_path / f"{v_tag}.parquet"
    rich.print(f"{v_fpath=}")
    if not v_fpath.exists():
        rich.print(f"{v_fpath} does not exist")
        return
    df_v = pd.read_parquet(v_fpath)

    u_tag = f"usc-{congress_num}-unified-v1"
    u_fpath = congress_hf_path / f"{u_tag}.parquet"
    rich.print(f"{u_fpath=}")
    if not u_fpath.exists():
        rich.print(f"{u_fpath} does not exist")
        return
    df_u = pd.read_parquet(u_fpath)
    df_u = df_u.rename(columns={"latest_text_id": "text_id"})
    df_u["metadata"] = df_u.apply(metadata_from_unified_row, axis=1)

    # take metadata from df_u
    df_out = pd.merge(
        df_v[["chunk_id", "text_id", "legis_id", "text", "vec"]],
        df_u[["text_id", "metadata"]],
        on="text_id",
    )
    cols = ["chunk_id", "text_id", "legis_id", "text", "metadata", "vec"]
    df_out = df_out[cols]

    with tempfile.TemporaryFile() as fp:
        df_out.to_parquet(fp)

        api = HfApi()
        repo_id = f"hyperdemocracy/{v_tag}"
        rich.print(f"{repo_id=}")
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            exist_ok=True,
        )
        api.upload_file(
            path_or_fileobj=fp,
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
    for model_name in model_names[0:1]:
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
