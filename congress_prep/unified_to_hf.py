"""
Gather all text versions for a given bill status row.
Create plain text from xml
"""

from collections import Counter
import datetime
import json
from typing import Union

from huggingface_hub import HfApi
from pathlib import Path
import pandas as pd
import rich

from bill_status_mod import BillStatus
from textversions_mod import get_bill_text_v1


def write_local(congress_hf_path: Union[str, Path], congress_num: int):

    congress_hf_path = Path(congress_hf_path)
    rich.print(f"{congress_hf_path=}")
    rich.print(f"{congress_num=}")

    df_bs = pd.read_parquet(
        congress_hf_path / f"usc-{congress_num}-billstatus-parsed.parquet"
    )
    assert df_bs["legis_id"].nunique() == df_bs.shape[0]

    df_tvs = pd.read_parquet(
        congress_hf_path / f"usc-{congress_num}-textversions-ddt-xml.parquet"
    )
    assert df_tvs["text_id"].nunique() == df_tvs.shape[0]
    missing_tvs = Counter()

    dfu = []

    for _, bs_row in df_bs.iterrows():

        billstatus_xml = bs_row["xml"]
        bs = BillStatus.from_xml_str(billstatus_xml)

        # find corresponding text versions
        tvs = []
        for bs_tv in bs.text_versions:

            # the file name in the url is the join key (e.g. BILLS-113hconres1rds.xml)
            if bs_tv.url:
                tv_url = bs_tv.url
                url_fname = Path(tv_url).name
            else:
                continue

            df_tv = df_tvs[df_tvs["file_name"] == url_fname]
            if df_tv.shape[0] > 1:
                raise ValueError()

            if df_tv.shape[0] == 0:
                missing_tvs[tv_url] += 1
                print(f"missing {bs_tv=}")
                continue

            # combine info from bill status and text version
            bs_tv = json.loads(bs_tv.model_dump_json())
            tv = df_tv.iloc[0].to_dict()
            tv["bs_date"] = bs_tv["date"]
            tv["bs_type"] = bs_tv["type"]
            tv["url"] = bs_tv["url"]

            xml = tv["xml"]
            xml = xml.strip()
            xml = xml.replace(" & ", " &amp; ")
            tv["text_v1"] = get_bill_text_v1(xml)
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

        urow = bs_row.to_dict()
        urow["text_versions"] = tvs
        urow["latest_text_id"] = (
            urow["text_versions"][0]["text_id"]
            if len(urow["text_versions"]) > 0
            else None
        )
        dfu.append(urow)

    dfu = pd.DataFrame(dfu)
    col = dfu.pop("latest_text_id")
    dfu.insert(1, col.name, col)

    fout = congress_hf_path / f"usc-{congress_num}-unified-v1.parquet"
    dfu.to_parquet(fout)


def upload_hf(congress_hf_path: Union[str, Path], congress_num: int):

    congress_hf_path = Path(congress_hf_path)
    rich.print(f"{congress_hf_path=}")
    rich.print(f"{congress_num=}")

    tag = f"usc-{congress_num}-unified-v1"
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
            path_or_fileobj=fpath,
            path_in_repo=fpath.name,
            repo_id=repo_id,
            repo_type="dataset",
        )


if __name__ == "__main__":

    congress_hf_path = Path("/Users/galtay/data/congress-hf")
    congress_nums = [113, 114, 115, 116, 117, 118]
    for congress_num in congress_nums:
        #        write_local(congress_hf_path, congress_num)
        upload_hf(congress_hf_path, congress_num)
