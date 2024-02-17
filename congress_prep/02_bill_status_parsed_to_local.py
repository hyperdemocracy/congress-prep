import json
from pathlib import Path
from typing import Union

from huggingface_hub import HfApi
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import rich

from congress_prep.bill_status_mod import BillStatus


def write_local(
    congress_hf_path: Union[str, Path],
    congress_nums: list[int],
):
    """Write local parquet file

    congress_hf_path: directory for local parquet files
    congress_nums: congress nums
    """

    congress_hf_path = Path(congress_hf_path)
    rich.print(f"{congress_hf_path=}")
    rich.print(f"{congress_nums=}")

    xml_file_paths = [
        congress_hf_path / f"usc-{cn}-billstatus-xml.parquet"
        for cn in congress_nums
    ]
    df = pd.concat([
        pd.read_parquet(xml_file_path) for xml_file_path in xml_file_paths
        if xml_file_path.exists()
    ]).reset_index(drop=True)

    bss = []
    for _, row in df.iterrows():
        bs = BillStatus.from_xml_str(row["xml"])
        # make everything serializable
        bss.append(json.loads(bs.model_dump_json()))
    df_bss = pd.DataFrame(bss)

    df["metadata"] = bss
    df = df.drop(columns=["xml"])

    table = pa.Table.from_pandas(df)
    for cn in df["congress_num"].unique():
        tf = table.filter((df['congress_num']==cn).values)
        fpath = congress_hf_path / f"usc-{cn}-billstatus-parsed.parquet"
        rich.print(f"{fpath=}")
        pq.write_table(tf, fpath)


if __name__ == "__main__":

    congress_hf_path = Path("/Users/galtay/data/congress-hf")
    congress_nums = list(range(108, 119))
    write_local(congress_hf_path, congress_nums)
