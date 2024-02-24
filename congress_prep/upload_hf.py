"""
Upload local data to HF
"""

from pathlib import Path
from typing import Optional, Union

from huggingface_hub import HfApi
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import rich


def upload_hf(
    congress_hf_path: Union[str, Path], congress_nums: list[int], file_types: list[str]
):
    """Upload local files to huggingface

    congress_hf_path: directory storing local parquet files
    congress_nums: list of congress numbers e.g. [113, 114, 115]
    file_types: list of file types to upload
    """

    congress_hf_path = Path(congress_hf_path)
    api = HfApi()

    rich.print(f"{congress_hf_path=}")
    rich.print(f"{congress_nums=}")
    rich.print(f"{file_types=}")

    repo_id = f"hyperdemocracy/us-congress"
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        exist_ok=True,
    )

    fpath = congress_hf_path / "README.md"
    rich.print(f"{fpath=}")
    api.upload_file(
        path_or_fileobj=fpath,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )

    for cn in congress_nums:

        rich.print(f"congress_num={cn}")

        # upload billstatus xml files
        # --------------------------------
        file_type = "billstatus_xml"
        if file_type in file_types:
            upload_folder = congress_hf_path / "usc-billstatus-xml"
            rich.print(f"{upload_folder=}")
            api.upload_folder(
                folder_path=upload_folder,
                path_in_repo=str(Path("data") / file_type),
                repo_id=repo_id,
                repo_type="dataset",
            )

        # upload textversions xml files
        # --------------------------------
        for xml_type in ["dtd_xml", "uslm_xml"]:
            xml_tag = xml_type.replace("_", "-")
            file_type = f"textversions_{xml_type}"
            if file_type in file_types:
                upload_folder = congress_hf_path / f"usc-textversions-{xml_tag}"
                rich.print(f"{upload_folder=}")
                api.upload_folder(
                    folder_path=upload_folder,
                    path_in_repo=str(Path("data") / file_type),
                    repo_id=repo_id,
                    repo_type="dataset",
                )

        # upload billstatus parsed files
        # --------------------------------
        file_type = "billstatus_parsed"
        if file_type in file_types:
            upload_folder = congress_hf_path / "usc-billstatus-parsed"
            rich.print(f"{upload_folder=}")
            api.upload_folder(
                folder_path=upload_folder,
                path_in_repo=str(Path("data") / file_type),
                repo_id=repo_id,
                repo_type="dataset",
            )

        # upload unified v1 files
        # --------------------------------
        file_type = "unified_v1"
        if file_type in file_types:
            upload_folder = congress_hf_path / "usc-unified-v1"
            rich.print(f"{upload_folder=}")
            api.upload_folder(
                folder_path=upload_folder,
                path_in_repo=str(Path("data") / file_type),
                repo_id=repo_id,
                repo_type="dataset",
            )

        # upload chunking files
        # --------------------------------
        fts = [
            "chunks_v1_s1024_o256",
            "chunks_v1_s2048_o256",
            "chunks_v1_s4096_o512",
            "chunks_v1_s8192_o512",
        ]
        for file_type in fts:
            if file_type in file_types:
                chunk_tag = file_type.replace("_", "-")
                upload_folder = congress_hf_path / f"usc-{chunk_tag}"
                api.upload_folder(
                    folder_path=upload_folder,
                    path_in_repo=str(Path("data") / file_type),
                    repo_id=repo_id,
                    repo_type="dataset",
                )


if __name__ == "__main__":

    congress_hf_path = Path("/Users/galtay/data/congress-hf")

    do_meta = True
    do_text = True

    if do_meta:
        file_types = [
            "billstatus_xml",
            "textversions_dtd_xml",
            "textversions_uslm_xml",
            "billstatus_parsed",
        ]
        congress_nums = list(range(108, 119))
        upload_hf(congress_hf_path, congress_nums, file_types)

    if do_text:
        file_types = [
            "unified_v1",
            "chunks_v1_s1024_o256",
            "chunks_v1_s2048_o256",
            "chunks_v1_s4096_o512",
            "chunks_v1_s8192_o512",
        ]
        congress_nums = list(range(113, 119))
        upload_hf(congress_hf_path, congress_nums, file_types)
