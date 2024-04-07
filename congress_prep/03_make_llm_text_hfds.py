from datasets import load_dataset
from datasets import concatenate_datasets
from datasets import DatasetDict


def create_text_col(sample):
    sample["text"] = (
        sample["tvs"][0]["tv_txt"].strip() if len(sample["tvs"]) > 0 else ""
    )
    return sample


def create_url_col(sample):
    sample["url"] = sample["tvs"][0]["bs_tv"]["url"] if len(sample["tvs"]) > 0 else ""
    return sample


def add_cols(sample):
    sample = create_text_col(sample)
    sample = create_url_col(sample)
    return sample


if __name__ == "__main__":

    dsd_orig = load_dataset("hyperdemocracy/usc-unified")
    ds_train = concatenate_datasets(
        [dsd_orig[cn] for cn in ["113", "114", "115", "116", "117"]]
    )

    ds_validation = dsd_orig["118"].filter(lambda x, idx: idx%2==0, with_indices=True)
    ds_test = dsd_orig["118"].filter(lambda x, idx: idx%2==1, with_indices=True)
    dsd = DatasetDict(
        {
            "train": ds_train,
            "validation": ds_validation,
            "test": ds_test,
        }
    )

    dsd = dsd.map(add_cols)
    columns_to_keep = [
        "legis_id",
        "url",
        "text",
    ]
    columns_to_remove = [
        col for col in dsd["train"].column_names if col not in columns_to_keep
    ]
    dsd = dsd.remove_columns(columns_to_remove)
    dsd = dsd.filter(lambda x: len(x["text"]) > 0)

    dsd.push_to_hub(f"hyperdemocracy/usc-llm-text")
