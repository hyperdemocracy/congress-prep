"""
Write scrapes from https://github.com/unitedstates/congress to local parquet
"""

from collections import Counter
from pathlib import Path
import re
from typing import Optional, Union

from huggingface_hub import HfApi
import pandas as pd
import rich


BILLS_PATTERN = re.compile(r"BILLS-(\d{3})([a-zA-Z]+)(\d+)(\w+)\.xml")
PLAW_PATTERN = re.compile(r"PLAW-(\d{3})([a-zA-Z]+)(\d+)\.xml")
FILE_PATTERN = re.compile(
    r"data/(\d{3})/(\w+)/(\w+)/([a-zA-Z]+)(\d+)/fdsys_billstatus\.xml"
)


def dataframe_from_scrape_files(
    congress_scraper_path: Union[str, Path]
) -> pd.DataFrame:
    """Read all scraped XML files into a DataFrame

    congress_scraper_path: scraper output directory. should contain "data" and "cache" as sub-directories
    """

    data_path = Path(congress_scraper_path) / "data"
    names = Counter()

    rows = []
    for path_object in data_path.rglob("*"):
        if path_object.suffix != ".xml":
            continue

        path_str = str(path_object.relative_to(congress_scraper_path))
        if path_object.name == "fdsys_billstatus.xml":
            file_type = "billstatus"
            legis_version = None
            match = re.match(FILE_PATTERN, path_str)
            if match:
                congress_num, legis_class, legis_type, _, legis_num = match.groups()
            else:
                print("billstatus oops: {}".format(path_object))
                continue

        else:
            if "/uslm/" in path_str:
                file_type = "uslm_xml"
            else:
                file_type = "ddt_xml"

            match = re.match(BILLS_PATTERN, path_object.name)
            if match:
                legis_class = "BILLS"
                congress_num, legis_type, legis_num, legis_version = match.groups()
            else:
                match = re.match(PLAW_PATTERN, path_object.name)
                if match:
                    legis_class = "PLAW"
                    legis_version = None
                    congress_num, legis_type, legis_num = match.groups()
                    if match is None:
                        print("text oops: {}".format(path_object))
                        break

        congress_num = int(congress_num)
        legis_num = int(legis_num)

        metadata = {
            "legis_id": "{}-{}-{}".format(congress_num, legis_type, legis_num),
            "congress_num": congress_num,
            "legis_type": legis_type.lower(),
            "legis_num": legis_num,
            "legis_version": legis_version,
            "legis_class": legis_class.lower(),
            "path": path_str,
            "file_name": Path(path_str).name,
            "file_type": file_type,
            "xml": path_object.read_text(),
        }
        rows.append(metadata)
        names[path_object.name] += 1

    df = pd.DataFrame(rows)
    return df


def write_local(congress_scraper_path: Union[str, Path]):
    """Write local parquet files

    congress_scraper_path: scraper output directory. should contain "data" and "cache" as sub-directories
    """

    congress_scraper_path = Path(congress_scraper_path)
    congress_hf_path = congress_scraper_path.parent / "congress-hf"
    df_all = dataframe_from_scrape_files(congress_scraper_path)

    for cn, df_cn in df_all.groupby("congress_num"):

        rich.print(f"congress_num={cn}")

        # write billstatus dataset
        # --------------------------------
        df_out = df_cn[df_cn["file_type"] == "billstatus"]
        if df_out.shape[0] > 0:
            cols = [
                "legis_id",
                "congress_num",
                "legis_type",
                "legis_num",
                "path",
                "xml",
            ]
            fpath = congress_hf_path / f"usc-{cn}-billstatus-xml.parquet"
            df_out = df_out.sort_values(["legis_type", "legis_num"]).reset_index(
                drop=True
            )
            df_out = df_out[cols]
            rich.print(fpath)
            df_out.to_parquet(fpath)

        # write xml textversions dataset
        # --------------------------------
        for xml_type in ["ddt_xml", "uslm_xml"]:
            df_out = df_cn[df_cn["file_type"] == xml_type].copy()
            if df_out.shape[0] > 0:
                cols = [
                    "text_id",
                    "legis_id",
                    "congress_num",
                    "legis_type",
                    "legis_num",
                    "legis_version",
                    "legis_class",
                    "path",
                    "file_name",
                    "xml",
                ]
                df_out["text_id"] = df_out.apply(
                    lambda x: x["legis_id"] + "-" + str(x["legis_version"]), axis=1
                )
                assert df_out["text_id"].nunique() == df_out.shape[0]
                fpath = congress_hf_path / "usc-{}-textversions-{}.parquet".format(
                    cn, xml_type.replace("_", "-")
                )
                df_out = df_out.sort_values(
                    ["legis_type", "legis_num", "legis_version"]
                ).reset_index(drop=True)
                df_out = df_out[cols]
                rich.print(fpath)
                df_out.to_parquet(fpath)


if __name__ == "__main__":

    congress_scraper_path = Path("/Users/galtay/data/congress-scraper")

    # to just read the dataframe
#    df = dataframe_from_scrape_files(congress_scraper_path)

# to write local parquet files
#    write_local(congress_scraper_path)

