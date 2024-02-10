import json
from pathlib import Path

from datasets import load_dataset
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import rich
import pandas as pd


def split_dataset(cn: int, chunk_size: int, chunk_overlap: int, use_local_data=True):

    if use_local_data:
        congress_hf_path = Path("/Users/galtay/data/congress-hf")
        df_path = congress_hf_path / f"usc-{cn}-unified-v1.parquet"
        rich.print(df_path)
        df_in = pd.read_parquet(df_path)
    else:
        ds_name = f"hyperdemocracy/usc-{cn}-unified-v1"
        rich.print(ds_name)
        ds = load_dataset(ds_name, split="train")
        df_in = ds.to_pandas()

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
                "origin_chamber": sample["origin_chamber"],
                "type": sample["type"],
                "update_date": sample["update_date"],
                "text_date": sample["text_versions"][0]["bs_date"],
                "introduced_date": sample["introduced_date"],
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
        #        dd = json.dumps({
        #            "page_content": doc.page_content,
        #            "metadata": doc.metadata,
        #        })
        #        fp.write(f"{dd}\n")
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

    fout = congress_hf_path / f"usc-{cn}-chunks-v1-s{chunk_size}-o{chunk_overlap}.parquet"
    df_out.to_parquet(fout)


if __name__ == "__main__":
    """
    Expected IDs to be unique, found duplicates of: 113-publ-76-None-565374, 113-publ-76-None-803979
    """

    cns = [113, 114, 115, 116, 117, 118]
    chunk_size = 1024
    chunk_overlap = 256

    for cn in cns:
        split_dataset(cn, chunk_size, chunk_overlap)
