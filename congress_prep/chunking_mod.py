import json
from pathlib import Path
from typing import Union
import tempfile

from datasets import load_dataset
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from huggingface_hub import HfApi
import rich
import pandas as pd


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
):

    congress_hf_path = Path(congress_hf_path)
    rich.print(f"{congress_hf_path=}")
    rich.print(f"{congress_num=}")
    rich.print(f"{chunk_size=}")
    rich.print(f"{chunk_overlap=}")

    df_path = congress_hf_path / f"usc-{congress_num}-unified-v1.parquet"
    rich.print(df_path)
    df_in = pd.read_parquet(df_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        add_start_index=True,
    )

    skipped = []
    docs = []
    for _, sample in df_in.iterrows():
        if len(sample["text_versions"]) == 0:
            skipped.append(sample["legis_id"])
            continue
        doc = Document(
            page_content=sample["text_versions"][0]["text_v1"],
            metadata={
                "text_id": sample["text_versions"][0]["text_id"],
                "legis_version": sample["text_versions"][0]["legis_version"],
                "legis_class": sample["text_versions"][0]["legis_class"],
                "legis_id": sample["legis_id"],
                "congress_num": sample["congress_num"],
                "legis_type": sample["legis_type"],
                "legis_num": sample["legis_num"],
                "text_date": sample["text_versions"][0]["bs_date"],
            },
        )
        docs.append(doc)

    split_docs = text_splitter.split_documents(docs)

    chunk_indx = -1
    cur_text_id = split_docs[0].metadata["text_id"]
    for doc in split_docs:

        if cur_text_id == doc.metadata["text_id"]:
            chunk_indx += 1
        else:
            chunk_indx = 0

        doc.metadata["chunk_id"] = "{}-{}".format(
            doc.metadata["text_id"],
            chunk_indx,
        )

        cur_text_id = doc.metadata["text_id"]

    df_out = pd.DataFrame.from_records(
        [
            {
                "chunk_id": doc.metadata["chunk_id"],
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in split_docs
        ]
    )

    tag = f"usc-{congress_num}-chunks-v1-s{chunk_size}-o{chunk_overlap}"
    fout = congress_hf_path / f"{tag}.parquet"
    df_out.to_parquet(fout)


def upload_hf(
    congress_hf_path: Union[str, Path],
    congress_num: int,
    chunk_size: int,
    chunk_overlap: int,
):

    congress_hf_path = Path(congress_hf_path)
    rich.print(f"{congress_hf_path=}")
    rich.print(f"{congress_num=}")
    rich.print(f"{chunk_size=}")
    rich.print(f"{chunk_overlap=}")

    c_tag = f"usc-{congress_num}-chunks-v1-s{chunk_size}-o{chunk_overlap}"
    c_fpath = congress_hf_path / f"{c_tag}.parquet"
    rich.print(f"{c_fpath=}")
    if not c_fpath.exists():
        rich.print(f"{c_fpath} does not exist")
        return
    df_c = pd.read_parquet(c_fpath)
    df_c["text_id"] = df_c["metadata"].apply(lambda x: x["text_id"])
    df_c["legis_id"] = df_c["metadata"].apply(lambda x: x["legis_id"])

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
        df_c[["chunk_id", "text_id", "legis_id", "page_content"]],
        df_u[["text_id", "metadata"]],
        on="text_id",
    )
    cols = ["chunk_id", "text_id", "legis_id", "page_content", "metadata"]
    df_out = df_out[cols]

    with tempfile.TemporaryFile() as fp:
        df_out.to_parquet(fp)

        api = HfApi()
        repo_id = f"hyperdemocracy/{c_tag}"
        rich.print(f"{repo_id=}")
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            exist_ok=True,
        )
        api.upload_file(
            path_or_fileobj=fp,
            path_in_repo=c_fpath.name,
            repo_id=repo_id,
            repo_type="dataset",
        )


if __name__ == "__main__":

    congress_hf_path = Path("/Users/galtay/data/congress-hf")
    chunk_size = 1024
    chunk_overlap = 256
    congress_nums = [113, 114, 115, 116, 117, 118]
    for congress_num in congress_nums:
        write_local(congress_hf_path, congress_num, chunk_size, chunk_overlap)
        upload_hf(congress_hf_path, congress_num, chunk_size, chunk_overlap)
