from datasets import load_dataset
from huggingface_hub import HfApi
from pathlib import Path
from nomic import atlas
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import rich
import yaml


vec_dtype = "float32"


CHUNK_METADATA_FEATURES = []


def get_readme_str_with_meta(chunk_subset: str, congress_nums: list[int]):
    chunk_tag = chunk_subset.replace("_", "-")
    yaml_dict = {
        "configs": [
            {
                "config_name": "default",
                "data_files": [
                    {
                        "split": cn,
                        "path": f"data/usc-{cn}-nomic-{chunk_tag}.parquet",
                    }
                    for cn in congress_nums
                ],
            }
        ],
        "dataset_info": {
            "features": [
                {"name": "chunk_id", "dtype": "string"},
                {"name": "congress_num", "dtype": "string"},
                {"name": "nomic_topic_depth_1", "dtype": "string"},
                {"name": "nomic_topic_depth_2", "dtype": "string"},
                {"name": "nomic_topic_depth_3", "dtype": "string"},
                {"name": "nomic_proj_x", "dtype": "float32"},
                {"name": "nomic_proj_y", "dtype": "float32"},
                {"name": "nomic_vec", "list": {"dtype": vec_dtype}},
                {"name": "text", "dtype": "string"},
                {
                    "name": "chunk_metadata",
                    "struct": [
                        {"name": "chunk_id", "dtype": "string"},
                        {"name": "chunk_index", "dtype": "int32"},
                        {"name": "congress_num", "dtype": "string"},
                        {"name": "legis_class", "dtype": "string"},
                        {"name": "legis_id", "dtype": "string"},
                        {"name": "legis_num", "dtype": "int32"},
                        {"name": "legis_type", "dtype": "string"},
                        {"name": "legis_version", "dtype": "string"},
                        {"name": "start_index", "dtype": "int32"},
                        {"name": "text_date", "dtype": "string"},
                        {"name": "text_id", "dtype": "string"},
                    ],
                },
                {
                    "name": "bill_metadata",
                    "struct": [
                        {"name": "introduced_date", "dtype": "string"},
                        {"name": "origin_chamber", "dtype": "string"},
                        {"name": "policy_area", "dtype": "string"},
                        {"name": "subjects", "list": {"dtype": "string"}},
                        {
                            "name": "sponsors",
                            "list": [
                                {"name": "bioguide_id", "dtype": "string"},
                                {"name": "district", "dtype": "string"},
                                {"name": "first_name", "dtype": "string"},
                                {"name": "full_name", "dtype": "string"},
                                {"name": "is_by_request", "dtype": "string"},
                                {"name": "last_name", "dtype": "string"},
                                {"name": "middle_name", "dtype": "string"},
                                {"name": "party", "dtype": "string"},
                                {"name": "state", "dtype": "string"},
                                {
                                    "name": "identifiers",
                                    "struct": [
                                        {"name": "bioguide_id", "dtype": "string"},
                                        {"name": "lis_id", "dtype": "string"},
                                        {"name": "gpo_id", "dtype": "string"},
                                    ],
                                },
                            ],
                        },
                    ],
                },
            ]
        },
    }

    readme_str = "---\n{}---".format(yaml.safe_dump(yaml_dict))
    return readme_str


def get_readme_str_no_meta(chunk_subset: str, congress_nums: list[int]):
    chunk_tag = chunk_subset.replace("_", "-")
    yaml_dict = {
        "configs": [
            {
                "config_name": "default",
                "data_files": [
                    {
                        "split": cn,
                        "path": f"data/usc-{cn}-nomic-no-meta-{chunk_tag}.parquet",
                    }
                    for cn in congress_nums
                ],
            }
        ],
        "dataset_info": {
            "features": [
                {"name": "chunk_id", "dtype": "string"},
                {"name": "congress_num", "dtype": "string"},
                {"name": "nomic_topic_depth_1", "dtype": "string"},
                {"name": "nomic_topic_depth_2", "dtype": "string"},
                {"name": "nomic_topic_depth_3", "dtype": "string"},
                {"name": "nomic_proj_x", "dtype": "float32"},
                {"name": "nomic_proj_y", "dtype": "float32"},
                {"name": "nomic_vec", "list": {"dtype": vec_dtype}},
            ]
        },
    }
    readme_str = "---\n{}---".format(yaml.safe_dump(yaml_dict))
    return readme_str


def write_local_no_meta(
    congress_nomic_path: Path, project_names: list[str], chunk_subsets: list[str]
):

    for project_name, chunk_subset in zip(project_names, chunk_subsets):

        chunk_tag = chunk_subset.replace("_", "-")
        tag = "usc-nomic-no-meta-{}".format(chunk_tag)
        repo_id = f"hyperdemocracy/{tag}"
        out_dir = congress_nomic_path / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        data_dir = out_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        print(f"{project_name=}")
        print(f"{chunk_subset=}")
        print(f"{out_dir=}")

        ds = atlas.AtlasDataset(identifier=project_name)
        map = ds.maps[0]

        # get low dimensional vecs
        df_vecs = map.embeddings.projected

        # add original vecs
        arr_latent_vecs = map.embeddings.latent
        df_vecs["nomic_vec"] = list(arr_latent_vecs)
        df_vecs["nomic_vec"] = df_vecs["nomic_vec"].apply(lambda x: x.astype(vec_dtype))

        # add topics
        df_topics = map.topics.df

        df_out = pd.merge(df_vecs, df_topics, on="chunk_id")
        df_out = df_out.rename(
            columns={
                "x": "nomic_proj_x",
                "y": "nomic_proj_y",
                "topic_depth_1": "nomic_topic_depth_1",
                "topic_depth_2": "nomic_topic_depth_2",
                "topic_depth_3": "nomic_topic_depth_3",
            }
        )
        df_out["congress_num"] = df_out["chunk_id"].apply(lambda x: x.split("-")[0])

        col_order = [
            "chunk_id",
            "congress_num",
            "nomic_topic_depth_1",
            "nomic_topic_depth_2",
            "nomic_topic_depth_3",
            "nomic_proj_x",
            "nomic_proj_y",
            "nomic_vec",
        ]
        df_out = df_out[col_order]

        congress_nums = sorted(df_out["congress_num"].unique())
        readme_str = get_readme_str_no_meta(chunk_subset, congress_nums)
        fpath = out_dir / "README.md"
        with open(fpath, "w") as fp:
            fp.write(readme_str)

        table = pa.Table.from_pandas(df_out)
        for cn in congress_nums:
            tf = table.filter((df_out["congress_num"] == cn).values)
            fpath = data_dir / "usc-{}-nomic-no-meta-{}.parquet".format(cn, chunk_tag)
            rich.print(f"{fpath=}")
            pq.write_table(tf, fpath)


def write_local_with_meta(
    congress_nomic_path: Path, project_names: list[str], chunk_subsets: list[str]
):

    for project_name, chunk_subset in zip(project_names, chunk_subsets):

        chunk_tag = chunk_subset.replace("_", "-")
        tag = "usc-nomic-{}".format(chunk_tag)
        repo_id = f"hyperdemocracy/{tag}"
        out_dir = congress_nomic_path / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        data_dir = out_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        print(f"{project_name=}")
        print(f"{chunk_subset=}")
        print(f"{out_dir=}")

        tag_no_meta = "usc-nomic-no-meta-{}".format(chunk_tag)
        df_nomic = pd.concat(
            [
                pd.read_parquet(
                    congress_nomic_path
                    / tag_no_meta
                    / "data"
                    / "usc-{}-nomic-no-meta-{}.parquet".format(cn, chunk_tag)
                )
                for cn in range(113, 119)
            ]
        )

        ds_chunks = load_dataset(
            "hyperdemocracy/us-congress", chunk_subset, split="all"
        )
        df_chunks = ds_chunks.to_pandas()
        df_chunks = df_chunks.rename(columns={"metadata": "chunk_metadata"})
        df_chunks = df_chunks.drop(columns=["text_id", "legis_id"])

        df_mrg = pd.merge(df_nomic, df_chunks, on="chunk_id")

        ds_meta = load_dataset(
            "hyperdemocracy/us-congress", "billstatus_parsed", split="all"
        )
        df_meta = ds_meta.to_pandas()
        df_meta = df_meta.rename(columns={"metadata": "bill_metadata"})

        keys = [
            "introduced_date",
            "origin_chamber",
            "policy_area",
            "subjects",
            "sponsors",
        ]
        df_mrg["bill_metadata"] = df_meta["bill_metadata"].apply(
            lambda x: {k: x[k] for k in keys}
        )

        # df_mrg = pd.merge(df_chunks, df_meta, on="legis_id")

        # assert df_chunks.shape[0] == df_mrg.shape[0] == df_nomic.shape[0]

        # sys.exit(0)

        # df_out = pd.merge(df_vecs, df_topics, on="chunk_id")
        # df_out = df_out.rename(
        #     columns={
        #         "x": "nomic_proj_x",
        #         "y": "nomic_proj_y",
        #         "topic_depth_1": "nomic_topic_depth_1",
        #         "topic_depth_2": "nomic_topic_depth_2",
        #         "topic_depth_3": "nomic_topic_depth_3",
        #     }
        # )
        # df_out = pd.merge(df_out, df_mrg, on="chunk_id")

        # col_order = [
        #     "chunk_id",
        #     "text_id",
        #     "legis_id",
        #     "congress_num",
        #     "legis_type",
        #     "legis_num",
        #     "nomic_topic_depth_1",
        #     "nomic_topic_depth_2",
        #     "nomic_topic_depth_3",
        #     "nomic_proj_x",
        #     "nomic_proj_y",
        #     "text",
        #     "chunk_metadata",
        #     "bill_metadata",
        #     "nomic_vec",
        # ]
        # df_out = df_out[col_order]

        df_out = df_mrg

        congress_nums = sorted(df_out["congress_num"].unique())
        readme_str = get_readme_str_with_meta(chunk_subset, congress_nums)
        fpath = out_dir / "README.md"
        with open(fpath, "w") as fp:
            fp.write(readme_str)

        table = pa.Table.from_pandas(df_out)
        for cn in congress_nums:
            tf = table.filter((df_out["congress_num"] == cn).values)
            fpath = data_dir / "usc-{}-nomic-{}.parquet".format(cn, chunk_tag)
            rich.print(f"{fpath=}")
            pq.write_table(tf, fpath)


def upload_hf_no_meta(
    congress_nomic_path: Path, project_names: list[str], chunk_subsets: list[str]
):

    api = HfApi()
    for project_name, chunk_subset in zip(project_names, chunk_subsets):

        chunk_tag = chunk_subset.replace("_", "-")
        tag = "usc-nomic-no-meta-{}".format(chunk_tag)
        repo_id = f"hyperdemocracy/{tag}"
        rich.print(repo_id)

        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            exist_ok=True,
        )

        fpath = congress_nomic_path / tag / "README.md"
        rich.print(f"{fpath=}")
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )

        for cn in range(113, 119):
            fpath = (
                congress_nomic_path
                / tag
                / "data"
                / "usc-{}-nomic-no-meta-{}.parquet".format(cn, chunk_tag)
            )
            if fpath.exists():
                rich.print(f"{fpath=}")
                api.upload_file(
                    path_or_fileobj=fpath,
                    path_in_repo=str(Path("data") / fpath.name),
                    repo_id=repo_id,
                    repo_type="dataset",
                )


def upload_hf_with_meta(
    congress_nomic_path: Path, project_names: list[str], chunk_subsets: list[str]
):

    api = HfApi()
    for project_name, chunk_subset in zip(project_names, chunk_subsets):

        chunk_tag = chunk_subset.replace("_", "-")
        tag = "usc-nomic-{}".format(chunk_tag)
        repo_id = f"hyperdemocracy/{tag}"
        rich.print(repo_id)

        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            exist_ok=True,
        )

        fpath = congress_nomic_path / tag / "README.md"
        rich.print(f"{fpath=}")
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )

        for cn in range(113, 119):
            fpath = (
                congress_nomic_path
                / tag
                / "data"
                / "usc-{}-nomic-{}.parquet".format(cn, chunk_tag)
            )
            if fpath.exists():
                rich.print(f"{fpath=}")
                api.upload_file(
                    path_or_fileobj=fpath,
                    path_in_repo=str(Path("data") / fpath.name),
                    repo_id=repo_id,
                    repo_type="dataset",
                )


project_names = [
    "gabrielhyperdemocracy/us-congressional-legislation-s8192o512nomic",
    "gabrielhyperdemocracy/us-congressional-legislation-s4096o512nomic",
    "gabrielhyperdemocracy/us-congressional-legislation-s2048o256nomic",
    "gabrielhyperdemocracy/us-congressional-legislation-s1024o256nomic",
][:1]

chunk_subsets = [
    "chunks_v1_s8192_o512",
    "chunks_v1_s4096_o512",
    "chunks_v1_s2048_o256",
    "chunks_v1_s1024_o256",
][:1]

congress_nomic_path = Path.home() / "data" / "congress-nomic"
# write_local_no_meta(congress_nomic_path, project_names, chunk_subsets)
# upload_hf_no_meta(congress_nomic_path, project_names, chunk_subsets)

write_local_with_meta(congress_nomic_path, project_names, chunk_subsets)
upload_hf_with_meta(congress_nomic_path, project_names, chunk_subsets)
