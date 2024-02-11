from pathlib import Path
import pandas as pd


congress_hf_path = Path("/Users/galtay/data/congress-hf")

cn = 113
ds_paths = {
    "bs_xml": congress_hf_path / f"usc-{cn}-billstatus-xml.parquet",
    "bs_parsed": congress_hf_path / f"usc-{cn}-billstatus-parsed.parquet",
    "tv_ddt_xml": congress_hf_path / f"usc-{cn}-textversions-ddt-xml.parquet",
    "tv_uslm_xml": congress_hf_path / f"usc-{cn}-textversions-uslm-xml.parquet",
    "unified_v1": congress_hf_path / f"usc-{cn}-unified-v1.parquet",
    "chunks_v1": congress_hf_path / f"usc-{cn}-chunks-v1-s1024-o256.parquet",
    "vecs_v1": congress_hf_path / f"usc-{cn}-vecs-v1-s1024-o256-BAAI-bge-small-en-v1.5.parquet",
}


dfs = {
    k: pd.read_parquet(v) for k,v in ds_paths.items()
}
