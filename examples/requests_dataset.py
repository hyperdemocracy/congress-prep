"""
'https://datasets-server.huggingface.co/rows?dataset=hyperdemocracy%2Fusc-118-unified-v1&config=default&split=train&offset=0&length=100'

"""

import requests
from typing import Optional
import urllib


BASE_URL = "https://datasets-server.huggingface.co"


def request_ds_end_point(
    end_point: str,
    ds_name: str,
    config: Optional[str] = None,
    split: Optional[str] = None,
    offset: Optional[int] = None,
    length: Optional[int] = None,
    base_url: str = BASE_URL,
):

    params = {"dataset": ds_name}
    if config:
        params["config"] = config
    if split:
        params["split"] = split
    if offset:
        params["offset"] = offset
    if length:
        params["length"] = length
    return requests.get(f"{base_url}/{end_point}", params=params).json()


ds_name = "hyperdemocracy/usc-118-unified-v1"
config = "default"
split = "train"

r_is_valid = request_ds_end_point("is-valid", ds_name)
r_splits = request_ds_end_point("splits", ds_name)

# returns first 100 rows of dataset
r_first_rows = request_ds_end_point("first-rows", ds_name, config=config, split=split)
features = r_first_rows["features"]
first_row = r_first_rows["rows"][0]

# index into dataset with offset and length
offset = 1000
length = 10  # max 100
r_rows = request_ds_end_point(
    "rows", ds_name, config=config, split=split, offset=offset, length=length
)




