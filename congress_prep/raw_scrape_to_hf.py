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


def dataframe_from_scrape(
    base_path: Union[str, Path],
    filter_congress_num: Optional[int] = None,
    filter_legis_type: Optional[str] = None,
) -> pd.DataFrame:
    """Read all scrape files into a DataFrame

    base_path: scraper output directory. should contain "data" and "cache" as subdirectories
    filter_congress_num: filter to one congress num
    """
    data_path = Path(base_path) / "data"
    names = Counter()

    rows = []
    for path_object in data_path.rglob("*"):
        if path_object.suffix == ".xml":
            path_str = str(path_object.relative_to(base_path))
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

            if filter_congress_num is not None and congress_num != filter_congress_num:
                continue

            if filter_legis_type is not None and legis_type != filter_legis_type:
                continue

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


def upload_scrape_to_hf(
    base_path: Union[str, Path],
    filter_congress_num: Optional[int] = None,
    filter_legis_type: Optional[str] = None,
):

    base_path = Path(base_path)
    out_path = base_path.parent / "congress-hf"
    df_all = dataframe_from_scrape(
        base_path,
        filter_congress_num=filter_congress_num,
        filter_legis_type=filter_legis_type,
    )
    api = HfApi()
    for cn, df in df_all.groupby("congress_num"):

        rich.print(f"congress_num={cn}")

        # upload billstatus dataset
        # --------------------------------
        df_out = df[df["file_type"] == "billstatus"]
        cols = ["legis_id", "congress_num", "legis_type", "legis_num", "path", "xml"]
        fout = out_path / f"usc-{cn}-billstatus.parquet"
        df_out = df_out.sort_values(["legis_type", "legis_num"]).reset_index(drop=True)[
            cols
        ]
        df_out.to_parquet(fout)
        if df_out.shape[0] > 0:
            repo_id = f"hyperdemocracy/usc-{cn}-billstatus"
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

        # upload textversions dataset
        # --------------------------------
        df_out = df[df["file_type"] == "ddt_xml"]
        cols = [
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
        fout = out_path / f"usc-{cn}-textversions.parquet"
        df_out = df_out.sort_values(
            ["legis_type", "legis_num", "legis_version"]
        ).reset_index(drop=True)[cols]
        df_out.to_parquet(fout)
        if df_out.shape[0] > 0:
            repo_id = f"hyperdemocracy/usc-{cn}-textversions"
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
    filter_legis_type = None  # "sres"
    filter_congress_num = 113
    df = dataframe_from_scrape(base_path, filter_congress_num, filter_legis_type)

    #    dataframe_from_scrape(base_path, filter_congress_num, filter_legis_type)
    upload_scrape_to_hf(base_path, filter_congress_num, filter_legis_type)
