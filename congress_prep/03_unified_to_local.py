"""
Gather all text versions for a given bill status row.
Create plain text from xml
"""

from collections import Counter
import datetime
import json
from typing import Union

from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import rich
from tqdm import tqdm

from bill_status_mod import BillStatus
from textversions_mod import get_bill_text_v1


def merge_bs_tv(congress_hf_path: Union[str, Path], cn: int):

    df_bs = pd.read_parquet(
        congress_hf_path
        / "usc-billstatus-parsed"
        / f"usc-{cn}-billstatus-parsed.parquet"
    )
    assert df_bs["legis_id"].nunique() == df_bs.shape[0]

    df_tv_xml = pd.read_parquet(
        congress_hf_path
        / "usc-textversions-dtd-xml"
        / f"usc-{cn}-textversions-dtd-xml.parquet"
    )
    assert df_tv_xml["text_id"].nunique() == df_tv_xml.shape[0]

    df_tvs = []
    missing_tvs = Counter()
    for _, bs_row in tqdm(df_bs.iterrows(), total=df_bs.shape[0], desc=f"merging {cn}"):

        # find corresponding text versions
        tvs = []
        for bs_tv in bs_row["metadata"]["text_versions"]:

            # the file name in the url is the join key (e.g. BILLS-113hconres1rds.xml)
            if bs_tv["url"]:
                tv_url = bs_tv["url"]
                url_fname = Path(tv_url).name
            else:
                continue

            ser_tv = df_tv_xml[df_tv_xml["file_name"] == url_fname]
            if ser_tv.shape[0] > 1:
                raise ValueError()

            if ser_tv.shape[0] == 0:
                missing_tvs[tv_url] += 1
                print(f"missing {bs_tv=}")
                continue

            ser_tv = ser_tv.iloc[0]

            # combine info from bill status and text version

            # this is the information from the textversions xml
            tv = ser_tv.to_dict()

            # this is the information from the billstatus xml
            tv["bs_date"] = bs_tv["date"]
            tv["bs_type"] = bs_tv["type"]
            tv["bs_url"] = bs_tv["url"]

            xml = tv["xml"]
            xml = xml.strip()
            xml = xml.replace(" & ", " &amp; ")
            tv["text_v1"] = get_bill_text_v1(xml)
            tv.pop("xml")
            tvs.append(tv)

        # sort all text versions for a bill by date.
        # most recent in front of list and date=None at the end
        tvs = sorted(
            tvs,
            key=lambda x: (
                x["bs_date"]
                if x["bs_date"] is not None
                else datetime.datetime.min.isoformat()
            ),
            reverse=True,
        )

        df_tvs.append(
            {
                "legis_id": bs_row["legis_id"],
                "textversions": tvs,
                "latest_text_id": tvs[0]["text_id"] if len(tvs) > 0 else None,
            }
        )

    df_tvs = pd.DataFrame(df_tvs)
    df_out = pd.merge(df_bs, df_tvs, on="legis_id", how="left")
    cols = [
        "legis_id",
        "latest_text_id",
        "congress_num",
        "legis_type",
        "legis_num",
        "metadata",
        "textversions",
    ]
    df_out = df_out[cols]
    assert df_out["legis_id"].nunique() == df_bs["legis_id"].nunique()

    return df_out


def write_local(congress_hf_path: Union[str, Path], congress_nums: list[int]):

    congress_hf_path = Path(congress_hf_path)
    rich.print(f"{congress_hf_path=}")
    rich.print(f"{congress_nums=}")

    df_out = pd.concat(
        [merge_bs_tv(congress_hf_path, cn) for cn in congress_nums]
    ).reset_index(drop=True)

    table = pa.Table.from_pandas(df_out)
    out_path = congress_hf_path / "usc-unified-v1"
    out_path.mkdir(parents=True, exist_ok=True)
    for cn in df_out["congress_num"].unique():
        tf = table.filter((df_out["congress_num"] == cn).values)
        fpath = out_path / f"usc-{cn}-unified-v1.parquet"
        rich.print(f"{fpath=}")
        pq.write_table(tf, fpath)


if __name__ == "__main__":

    congress_hf_path = Path("/Users/galtay/data/congress-hf")
    congress_nums = [113, 114, 115, 116, 117, 118]
    write_local(congress_hf_path, congress_nums)
