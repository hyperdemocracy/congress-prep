import json
from pathlib import Path
from typing import Union

from huggingface_hub import HfApi
import pandas as pd
import rich

from congress_prep.bill_status_mod import BillStatus


def write_local(
    congress_hf_path: Union[str, Path],
    congress_num: int,
):
    """Write local parquet file

    congress_hf_path: directory for local parquet files
    congress_num: congress num
    """

    congress_hf_path = Path(congress_hf_path)
    rich.print(f"{congress_hf_path=}")
    rich.print(f"{congress_num=}")

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
    df_hf = df_hf.drop(columns=["xml"])

    if df_hf.shape[0] > 0:
        fout = congress_hf_path / f"usc-{congress_num}-billstatus-parsed.parquet"
        df_hf.to_parquet(fout)


if __name__ == "__main__":

    congress_hf_path = Path("/Users/galtay/data/congress-hf")
    for congress_num in range(109, 119):
        write_local(congress_hf_path, congress_num)
