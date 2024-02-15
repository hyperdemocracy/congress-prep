import json
from datasets import load_dataset
from datasets import concatenate_datasets
from nomic import atlas
import numpy as np
import pandas as pd
import rich


CONGRESS_GOV_TYPE_MAP = {
    "hconres": "house-concurrent-resolution",
    "hjres": "house-joint-resolution",
    "hr": "house-bill",
    "hres": "house-resolution",
    "s": "senate-bill",
    "sconres": "senate-concurrent-resolution",
    "sjres": "senate-joint-resolution",
    "sres": "senate-resolution",
}


def get_sponsor_url(bioguide_id):
    return f"https://bioguide.congress.gov/search/bio/{bioguide_id}"


def get_congress_gov_url(congress_num, legis_type, legis_num):
    lt = CONGRESS_GOV_TYPE_MAP[legis_type]
    return f"https://www.congress.gov/bill/{congress_num}th-congress/{lt}/{legis_num}"


congress_nums = [113, 114, 115, 116, 117, 118]
#congress_nums = [113]

confs = {
    "mark1": {
        "chunk_size": 1024,
        "chunk_overlap": 256,
        "use_vecs": True,
        "bge_size": "large",
        "tag": "s1024.o256.bge-large"
    },
    "mark2": {
        "chunk_size": 8192,
        "chunk_overlap": 512,
        "use_vecs": False,
        "bge_size": None,
        "tag": "s8192.o512.nomic"
    },
}

config_name = "mark2"
conf = confs[config_name]

chunk_size = conf["chunk_size"]
chunk_overlap = conf["chunk_overlap"]
use_vecs = conf["use_vecs"]
bge_size = conf["bge_size"]
tag = conf["tag"]

num_points = None
project_name = "US Congressional Legislation"


df_uni = pd.concat(
    [
        load_dataset(
            f"hyperdemocracy/usc-{cn}-unified-v1",
            split="train",
        ).to_pandas()
        for cn in congress_nums
    ]
)


if use_vecs:

    df_vecs = pd.concat(
        [
            load_dataset(
                f"hyperdemocracy/usc-{cn}-vecs-v1-s{chunk_size}-o{chunk_overlap}-BAAI-bge-{bge_size}-en-v1.5",
                split="train",
            ).to_pandas()
            for cn in congress_nums
        ]
    )
    df_mrg = pd.merge(
        df_vecs,
        df_uni,
        on="legis_id",
    )

else:

    df_text = pd.concat(
        [
            load_dataset(
                f"hyperdemocracy/usc-{cn}-chunks-v1-s{chunk_size}-o{chunk_overlap}",
                split="train",
            ).to_pandas()
            for cn in congress_nums
        ]
    )
    df_mrg = pd.merge(
        df_text,
        df_uni,
        on="legis_id",
    )


df_mrg["subjects"] = df_mrg["subjects"].apply(lambda x: " | ".join(sorted(x)))
df_mrg["sponsor_url"] = df_mrg["metadata"].apply(
    lambda x: get_sponsor_url(x["sponsor"])
)
df_mrg["sponsor_name"] = df_mrg["sponsors"].apply(lambda x: x[0].get("full_name", ""))
df_mrg["congress_gov_url"] = df_mrg.apply(
    lambda x: get_congress_gov_url(
        x["congress_num"],
        x["legis_type"],
        x["legis_num"],
    ),
    axis=1,
)


def get_sponsor_party(x):
    """get D from
    Rep. McGovern, James P. [D-MA-2]
    """
    between = x[x.index("[") + 1 : x.index("]")]
    pieces = between.split("-")
    return pieces[0]


def get_sponsor_state(x):
    """get MA from
    Rep. McGovern, James P. [D-MA-2]
    """
    between = x[x.index("[") + 1 : x.index("]")]
    pieces = between.split("-")
    return pieces[1]


df_mrg["sponsor_party"] = df_mrg["sponsor_name"].apply(get_sponsor_party)
df_mrg["sponsor_state"] = df_mrg["sponsor_name"].apply(get_sponsor_state)

# nomic does better with dates than datetimes
for col in ["update_date", "update_date_including_text"]:
    df_mrg[col] = df_mrg[col].apply(lambda x: x if x is None else x.split("T")[0])


"""
Replacing 132 null values for field policy_area with string 'null'. This behavior will change in a future version.
"""

keep_cols = [
    "chunk_id",
    "text_id",
    "legis_id",
    "congress_num",
    "legis_type",
    "legis_num",
    "title",
    "text",
    "congress_gov_url",
    "sponsor_name",
    "sponsor_url",
    "introduced_date",
    "update_date",
    "origin_chamber",
    "sponsor_party",
    "sponsor_state",
    "policy_area",
    "subjects",
]
if use_vecs:
    keep_cols += ["vec"]

df_mrg = df_mrg[keep_cols]
df_mrg = df_mrg.reset_index(drop=True)
if num_points is None:
    df = df_mrg
    if use_vecs:
        embeddings = np.array(df_mrg["vec"].to_list())
        df = df.drop(columns=["vec"])
else:
    df = df_mrg.head(num_points).reset_index(drop=True)
    if use_vecs:
        embeddings = np.array(df_mrg.head(num_points)["vec"].to_list())
        df = df.drop(columns=["vec"])

if use_vecs:
    project = atlas.map_data(
        data=df,
        embeddings=embeddings,
        identifier=project_name,
        description="Legislation from the 113th to 118th US Congress",
        id_field="chunk_id",
        topic_model={"build_topic_model": True, "topic_label_field": "text"},
    )
else:
    project = atlas.map_data(
        data=df,
        identifier=project_name,
        description=f"Legislation from the 113th to 118th US Congress ({tag})",
        id_field="chunk_id",
        topic_model={"build_topic_model": True, "topic_label_field": "text"},
        indexed_field="text",
    )
