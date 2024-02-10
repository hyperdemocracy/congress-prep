from pathlib import Path
import pandas as pd


congress_hf_path = Path("/Users/galtay/data/congress-hf")

cn = 113
ds_paths = {
    "bs_xml": congress_hf_path / f"usc-{cn}-billstatus.parquet",
    "bs_parsed": congress_hf_path / f"usc-{cn}-billstatus-parsed.parquet",
    "tv_xml": congress_hf_path / f"usc-{cn}-textversions.parquet",
    "unified_v1": congress_hf_path / f"usc-{cn}-unified-v1.parquet",
}


dfs = {
    k: pd.read_parquet(v) for k,v in ds_paths.items()
}
