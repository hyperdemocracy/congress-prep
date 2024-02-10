import json
from pathlib import Path
from typing import Union

from huggingface_hub import HfApi
import pandas as pd
import rich

from congress_prep.bill_status_mod import BillStatus


def write_local(
    base_path: Union[str, Path],
    congress_num: int,
):

    """Write local parquet file

    base_path: scraper output directory. should contain "data" and "cache" as sub-directories
    congress_num: congress num
    """

    base_path = Path(base_path)
    congress_hf_path = base_path.parent / "congress-hf"

    rich.print(f"{congress_num=}")
    rich.print(f"{congress_hf_path=}")

    bss = []
    xml_file_path = congress_hf_path / f"usc-{congress_num}-billstatus-xml.parquet"
    df_xml = pd.read_parquet(xml_file_path)
    for _, row in df_xml.iterrows():
        xml = row["xml"]
        bs = BillStatus.from_xml_str(xml)
        # make everything serializable
        bss.append(json.loads(bs.model_dump_json()))
    df_bss = pd.DataFrame(bss)
    df_hf = pd.concat([df_xml, df_bss], axis=1)

    if df_hf.shape[0] > 0:
        fout = congress_hf_path / f"usc-{congress_num}-billstatus-parsed.parquet"
        df_hf.to_parquet(fout)



def upload_hf(
    base_path: Union[str, Path],
    congress_num: int,
):
    """Upload parsed bill status files to huggingface

    base_path: scraper output directory. should contain "data" and "cache" as sub-directories
    congress_num: congress num
    """

    base_path = Path(base_path)
    congress_hf_path = base_path.parent / "congress-hf"

    rich.print(f"{congress_num=}")
    rich.print(f"{congress_hf_path=}")

    tag = f"usc-{congress_num}-billstatus-parsed"
    fpath = congress_hf_path / f"{tag}.parquet"
    if fpath.exists():

        api = HfApi()
        repo_id = f"hyperdemocracy/{tag}"
        rich.print(f"{repo_id=}")
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

    base_path = Path("/Users/galtay/data/congress-scraper")
    for congress_num in range(109, 119):
        write_local(base_path, congress_num)
#        upload_hf(base_path, congress_num)
