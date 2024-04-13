"""
https://voteview.com/articles/data_help_members
"""

from pathlib import Path
import json
from nomic import atlas
import numpy as np
import pandas as pd
import rich


PARTY_CODE_MAP = {
    100: 'Democratic Party',
    200: 'Republican Party',
     1: 'Federalist Party',
     13: 'Democratic-Republican Party',
     22: 'Adams Party',
     26: 'Anti Masonic Party',
     29: 'Whig Party',
     37: 'Constitutional Unionist Party',
     44: 'Nullifier Party',
     46: 'States Rights Party',
     108: 'Anti-Lecompton Democrats',
     112: 'Conservative Party',
     114: 'Readjuster Party',
     117: 'Readjuster Democrats',
     203: 'Unconditional Unionist Party',
     206: 'Unionist Party',
     208: 'Liberal Republican Party',
     213: 'Progressive Republican Party',
     300: 'Free Soil Party',
     310: 'American Party',
     326: 'National Greenbacker Party',
     340: 'Populist PARTY',
     347: 'Prohibitionist Party',
     354: 'Silver Republican Party',
     355: 'Union Labor Party',
     356: 'Union Labor Party',
     370: 'Progressive Party',
     380: 'Socialist Party',
     402: 'Liberal Party',
     403: 'Law and Order Party',
     522: 'American Labor Party',
     523: 'American Labor Party (La Guardia)',
     537: 'Farmer-Labor Party',
     555: 'Jackson Party',
     1060: 'Silver Party',
     1111: 'Liberty Party',
     1116: 'Conservative Republicans',
     1275: 'Anti-Jacksonians',
     1346: 'Jackson Republican',
     3333: 'Opposition Party',
     3334: 'Opposition Party (36th)',
     4000: 'Anti-Administration Party',
     4444: 'Constitutional Unionist Party',
     5000: 'Pro-Administration Party',
     6000: 'Crawford Federalist Party',
     7000: 'Jackson Federalist Party',
     7777: 'Crawford Republican Party',
     8000: 'Adams-Clay Federalist Party',
     8888: 'Adams-Clay Republican Party',
     328: 'Independent',
     329: 'Independent Democrat',
     331: 'Independent Republican',
     603: 'Independent Whig',
}


HS_COLS = [
    'nominate_dim1',
    'nominate_dim2',
    'nominate_log_likelihood',
    'nominate_geo_mean_probability',
    'nominate_number_of_votes',
    'nominate_number_of_errors',
    'conditional',
    'nokken_poole_dim1',
    'nokken_poole_dim2'
]


def get_sponsor_url(bioguide_id):
    return f"https://bioguide.congress.gov/search/bio/{bioguide_id}"


def read_voteview_hsall(drop_president=True):
    data_path = Path("/Users/galtay/data")
    df = pd.read_csv(data_path / "voteview" / "HSall_members.csv")
    df = df[df["chamber"]!="President"]
    for col in ['nominate_dim1', 'nominate_dim2']:
        df = df[~df[col].isna()]
    df = df.reset_index(drop=True)
    return df


df = read_voteview_hsall()
df["bioguide_url"] = df["bioguide_id"].apply(get_sponsor_url)
df["party_name"] = df["party_code"].apply(lambda x: PARTY_CODE_MAP.get(x))
df["row_id"] = df.apply(lambda x: "{}-{}-{}-{}".format(
    x["congress"],
    x["bioguide_id"],
    x["icpsr"],
    x["district_code"],
), axis=1)
assert df.shape[0] == df['row_id'].nunique()

embeddings = df[['nominate_dim1', 'nominate_dim2']].values
project_name = "US Legislators - voteview v1"

project = atlas.map_data(
    data=df,
    embeddings=embeddings,
    identifier=project_name,
    description="Legislators from the US Congress",
    id_field="row_id",
    projection=False,
    topic_model=False,
    duplicate_detection=False,
)
