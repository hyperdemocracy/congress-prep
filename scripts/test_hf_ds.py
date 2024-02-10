from datasets import load_dataset

cn = 113
ds_names = {
    "tv_xml": f"hyperdemocracy/usc-{cn}-textversions-xml",
    "bs_xml": f"hyperdemocracy/usc-{cn}-billstatus-xml",
    "bs_parsed": f"hyperdemocracy/usc-{cn}-billstatus-parsed",
    "unified_v1": f"hyperdemocracy/usc-{cn}-unified-v1",
}


dss = {
    k: load_dataset(v, split="train") for k,v in ds_names.items()
}
