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

from congress_prep import utils


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
            match = re.match(utils.FILE_PATTERN, path_str)
            if match:
                congress_num, legis_class, legis_type, _, legis_num = match.groups()
            else:
                print("billstatus oops: {}".format(path_object))
                continue

            lastmod_path = path_object.parent / "fdsys_billstatus-lastmod.txt"
            lastmod_str = lastmod_path.read_text()

        else:
            if "/uslm/" in path_str:
                file_type = "uslm_xml"
            else:
                file_type = "dtd_xml"

            match = re.match(utils.BILLS_PATTERN, path_object.name)
            if match:
                legis_class = "BILLS"
                congress_num, legis_type, legis_num, legis_version = match.groups()
                lastmod_path = path_object.parent / (path_object.name.split(".")[0] + "-lastmod.txt")
                lastmod_str = lastmod_path.read_text()
            else:
                match = re.match(utils.PLAW_PATTERN, path_object.name)
                if match:
                    legis_class = "PLAW"
                    legis_version = None
                    congress_num, legis_type, legis_num = match.groups()
                    if match is None:
                        print("text oops: {}".format(path_object))
                        break
                    lastmod_path = path_object.parent / (path_object.name.split(".")[0] + "-lastmod.txt")
                    lastmod_str = lastmod_path.read_text()

        congress_num = int(congress_num)
        legis_num = int(legis_num)

        metadata = {
            "legis_id": "{}-{}-{}".format(congress_num, legis_type, legis_num),
            "congress_num": congress_num,
            "legis_type": legis_type.lower(),
            "legis_num": legis_num,
            "legis_version": legis_version,
            "legis_class": legis_class.lower(),
            "scrape_path": path_str,
            "file_name": Path(path_str).name,
            "file_type": file_type,
            "lastmod": lastmod_str,
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
                "scrape_path",
                "lastmod",
                "xml",
            ]
            out_path = congress_hf_path / "usc-billstatus-xml"
            out_path.mkdir(parents=True, exist_ok=True)
            fpath = out_path / f"usc-{cn}-billstatus-xml.parquet"
            df_out = df_out.sort_values(["legis_type", "legis_num"]).reset_index(
                drop=True
            )
            df_out = df_out[cols]
            rich.print(fpath)
            df_out.to_parquet(fpath)

        # write xml textversions dataset
        # --------------------------------
        for xml_type in ["dtd_xml", "uslm_xml"]:
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
                    "scrape_path",
                    "file_name",
                    "lastmod",
                    "xml",
                ]
                df_out["text_id"] = df_out.apply(
                    lambda x: x["legis_id"] + "-" + str(x["legis_version"]), axis=1
                )
                assert df_out["text_id"].nunique() == df_out.shape[0]
                out_path = congress_hf_path / "usc-textversions-{}".format(
                    xml_type.replace("_", "-")
                )
                out_path.mkdir(parents=True, exist_ok=True)
                fpath = out_path / "usc-{}-textversions-{}.parquet".format(
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
    write_local(congress_scraper_path)
