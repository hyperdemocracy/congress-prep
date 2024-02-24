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

from congress_prep import utils


def get_langchain_docs_from_unified(df_u: pd.DataFrame) -> list[Document]:
    skipped = []
    docs = []
    for _, u_row in df_u.iterrows():
        if len(u_row["textversions"]) == 0:
            skipped.append(u_row["legis_id"])
            continue
        doc = Document(
            page_content=u_row["textversions"][0]["text_v1"],
            metadata={
                "text_id": u_row["textversions"][0]["text_id"],
                "legis_version": u_row["textversions"][0]["legis_version"],
                "legis_class": u_row["textversions"][0]["legis_class"],
                "legis_id": u_row["legis_id"],
                "congress_num": u_row["congress_num"],
                "legis_type": u_row["legis_type"],
                "legis_num": u_row["legis_num"],
                "text_date": u_row["textversions"][0]["bs_date"],
            },
        )
        docs.append(doc)
    rich.print(f"skipped {len(skipped)} rows with no text versions")
    return docs


def add_chunk_index(split_docs: list[Document]) -> list[Document]:
    chunk_index = -1
    cur_text_id = split_docs[0].metadata["text_id"]
    for doc in split_docs:
        if cur_text_id == doc.metadata["text_id"]:
            chunk_index += 1
        else:
            chunk_index = 0
        doc.metadata["chunk_index"] = chunk_index
        doc.metadata["chunk_id"] = "{}-{}".format(
            doc.metadata["text_id"],
            chunk_index,
        )
        cur_text_id = doc.metadata["text_id"]
    return split_docs


def write_local(
    congress_hf_path: Union[str, Path],
    congress_num: int,
    chunk_size: int,
    chunk_overlap: int,
):

    rich.print("CHUNKING (write local)")
    congress_hf_path = Path(congress_hf_path)
    rich.print(f"{congress_hf_path=}")
    rich.print(f"{congress_num=}")
    rich.print(f"{chunk_size=}")
    rich.print(f"{chunk_overlap=}")

    u_fpath = (
        congress_hf_path / "usc-unified-v1" / f"usc-{congress_num}-unified-v1.parquet"
    )
    rich.print(u_fpath)
    df_u = pd.read_parquet(u_fpath)

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", ";", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        add_start_index=True,
    )

    docs = get_langchain_docs_from_unified(df_u)
    split_docs = text_splitter.split_documents(docs)
    split_docs = add_chunk_index(split_docs)

    df_c = pd.DataFrame.from_records(
        [
            {
                "chunk_id": doc.metadata["chunk_id"],
                "text": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in split_docs
        ]
    )

    df_c["text_id"] = df_c["metadata"].apply(lambda x: x["text_id"])
    df_c["legis_id"] = df_c["metadata"].apply(lambda x: x["legis_id"])

    chunk_tag = f"chunks-v1-s{chunk_size}-o{chunk_overlap}"
    file_tag = f"usc-{congress_num}-{chunk_tag}"

    cols = ["chunk_id", "text_id", "legis_id", "text", "metadata"]
    df_c = df_c[cols]
    out_path = congress_hf_path / f"usc-{chunk_tag}"
    out_path.mkdir(parents=True, exist_ok=True)
    fout = out_path / f"{file_tag}.parquet"
    rich.print(f"{fout=}")
    df_c.to_parquet(fout)


if __name__ == "__main__":

    congress_hf_path = Path("/Users/galtay/data/congress-hf")
    chunks = [(8192, 512), (4096, 512), (2048, 256), (1024, 256)]
    for chunk_size, chunk_overlap in chunks:
        congress_nums = [113, 114, 115, 116, 117, 118]
        for congress_num in congress_nums:
            write_local(congress_hf_path, congress_num, chunk_size, chunk_overlap)
