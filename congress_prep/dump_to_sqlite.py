"""
https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_sql.html
"""
import json
import os
from pathlib import Path
import rich

import pandas as pd
from sqlalchemy import create_engine

from congress_prep import orm_mod
from congress_prep.bill_status_mod import BillStatus


path_to_db = "/Users/galtay/myrepos/congress_prep/tmp.db"
os.remove(path_to_db)
conn_str = f"sqlite:///{path_to_db}"
echo = False
engine = create_engine(conn_str, echo=echo)
orm_mod.Base.metadata.create_all(engine)


congress_hf_path = Path("/Users/galtay/data/congress-hf")
datasets = [
    "billstatus-parsed",
    "billstatus-xml",
    "textversions-dtd-xml",
    "textversions-uslm-xml",
]

for ds in datasets:

    table_name = ds.replace("-", "_")
    ds_dir = congress_hf_path / f"usc-{ds}"
    fpaths = sorted(list(ds_dir.glob("*.parquet")))
    for ii, fpath in enumerate(fpaths):
        rich.print(fpath)
        df = pd.read_parquet(fpath)
        if ds == "billstatus-parsed":
            df["billstatus_json"] = df["billstatus_json"].apply(
                lambda x: BillStatus(**x).model_dump_json()
            )
        df.to_sql(table_name, con=engine, index=False, if_exists="append")
