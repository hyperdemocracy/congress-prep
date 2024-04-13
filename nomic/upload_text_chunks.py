from pathlib import Path
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

HS_COLS = [
    'nominate_dim1',
    'nominate_dim2',
#    'nominate_log_likelihood',
#    'nominate_geo_mean_probability',
#    'nominate_number_of_votes',
#    'nominate_number_of_errors',
#    'conditional',
    'nokken_poole_dim1',
    'nokken_poole_dim2'
]

def get_sponsor_url(bioguide_id):
    return f"https://bioguide.congress.gov/search/bio/{bioguide_id}"


def get_congress_gov_url(congress_num, legis_type, legis_num):
    lt = CONGRESS_GOV_TYPE_MAP[legis_type]
    return f"https://www.congress.gov/bill/{congress_num}th-congress/{lt}/{legis_num}"


def read_hsall():
    data_path = Path("/Users/galtay/data")
    df_hs = pd.read_csv(data_path / "voteview" / "HSall_members.csv")
    return df_hs


df_hs = read_hsall()

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
    "mark3": {
        "chunk_size": 4096,
        "chunk_overlap": 512,
        "use_vecs": False,
        "bge_size": None,
        "tag": "s4096.o512.nomic"
    },
    "mark4": {
        "chunk_size": 2048,
        "chunk_overlap": 256,
        "use_vecs": False,
        "bge_size": None,
        "tag": "s2048.o256.nomic"
    },
    "mark5": {
        "chunk_size": 1024,
        "chunk_overlap": 256,
        "use_vecs": False,
        "bge_size": None,
        "tag": "s1024.o256.nomic"
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
project_name = f"US Congressional Legislation ({tag})"


df_uni = pd.concat(
    [
        load_dataset(
            path="hyperdemocracy/usc-unified",
            split=f"{cn}",
        ).to_pandas().reset_index(drop=True)
        for cn in congress_nums
    ]
)
rich.print("df_uni.shape: {}".format(df_uni.shape))


if use_vecs:

    df_vecs = pd.concat(
        [
            load_dataset(
                f"hyperdemocracy/usc-{cn}-vecs-v1-s{chunk_size}-o{chunk_overlap}-BAAI-bge-{bge_size}-en-v1.5",
                split="train",
            ).to_pandas().reset_index(drop=True)
            for cn in congress_nums
        ]
    )
    rich.print("df_vecs.shape: {}".format(df_vecs.shape))
    df_mrg = pd.merge(
        df_vecs,
        df_uni,
        on="legis_id",
    )

else:

    df_text = load_dataset(
        path=f"hyperdemocracy/usc-chunks-s{chunk_size}-o{chunk_overlap}",
        split="train",
    ).to_pandas().reset_index(drop=True)
    rich.print("df_text.shape: {}".format(df_text.shape))

    df_mrg = pd.merge(
        df_text.rename(columns={"metadata": "metadata_text"}),
        df_uni.rename(columns={"bs_json": "metadata_uni"}),
        on="legis_id",
    )

rich.print("df_mrg.shape: {}".format(df_mrg.shape))

df_mrg["subjects"] = df_mrg["metadata_uni"].apply(lambda x: " | ".join(sorted(x["subjects"])))
df_mrg["sponsor_url"] = df_mrg["metadata_uni"].apply(
    lambda x: get_sponsor_url(x["sponsors"][0]["bioguide_id"])
)
df_mrg["sponsor_name"] = df_mrg["metadata_uni"].apply(lambda x: x["sponsors"][0].get("full_name", ""))
df_mrg["congress_gov_url"] = df_mrg.apply(
    lambda x: get_congress_gov_url(
        x["congress_num"],
        x["legis_type"],
        x["legis_num"],
    ),
    axis=1,
)

for col in ["title", "policy_area", "origin_chamber"]:
    df_mrg[col] = df_mrg["metadata_uni"].apply(lambda x: x[col])



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
for col in ["update_date", "update_date_including_text", "introduced_date"]:
    df_mrg[col] = df_mrg["metadata_uni"].apply(lambda x: x[col] if x[col] is None else x[col].split("T")[0])

keep_cols = [
    "chunk_id",
    "tv_id",
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

rich.print("df.shape: {}".format(df.shape))


for cn in congress_nums:
    df1 = df[df['congress_num']==cn].copy()
    rich.print("df1.shape: {}".format(df1.shape[0]))


df_new = pd.DataFrame()
for cn in congress_nums:
    rich.print("cn={}".format(cn))
    df1 = df[df['congress_num']==cn].copy()
    rich.print("df1.shape: {}".format(df1.shape[0]))
    df_hs1 = df_hs[df_hs["congress"]==cn].copy()
    df_hs1 = df_hs1.drop_duplicates(subset="bioguide_id")
    rich.print("df_hs1.shape: {}".format(df_hs1.shape[0]))

    df1['bioguide_id'] = df1['sponsor_url'].apply(lambda x: x.split("/")[-1])
    df1 = pd.merge(df1, df_hs1[HS_COLS + ["bioguide_id"]], how='left', on="bioguide_id")
    rich.print("df1.shape: {}".format(df1.shape[0]))
    print()
    df_new = pd.concat([df_new, df1])


df_new = df_new.reset_index(drop=True)
rich.print("df_new.shape: {}".format(df_new.shape[0]))
assert df_new.shape[0] == df.shape[0]
sys.exit(0)

if use_vecs:
    project = atlas.map_data(
        data=df_new,
        embeddings=embeddings,
        identifier=project_name,
        description=f"Legislation from the 113th to 118th US Congress ({tag})",
        id_field="chunk_id",
        topic_model={"build_topic_model": True, "topic_label_field": "text"},
    )
else:
    project = atlas.map_data(
        data=df_new,
        identifier=project_name,
        description=f"Legislation from the 113th to 118th US Congress ({tag})",
        id_field="chunk_id",
        topic_model={"build_topic_model": True, "topic_label_field": "text"},
        indexed_field="text",
    )

