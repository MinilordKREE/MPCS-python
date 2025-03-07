import pandas as pd
import json
import numpy as np

# merged_corpora
def get_merged_corpora(file_pth):
    mc_df = pd.read_csv(file_pth)
    mc_df.insert(0, "statement_id", range(0, len(mc_df)))
    mc_list = [(i, item)  for i, item in enumerate(mc_df["sentence"].to_list())]
    return mc_list, mc_df

def get_reddits(file_pth, mode = "train", dimension = "None"):
    reds_df = pd.read_csv(file_pth).iloc[34262:52148]
    if mode == "test":
        reds_df = reds_df[~reds_df[["fairness_train", "authority_train", "care_train", "loyalty_train", "sanctity_train"]].eq(1).all(axis=1)]
    if dimension is not None:
        reds_df = reds_df[reds_df[f"{dimension}_train"] == 0]
    reds_df.insert(0, "statement_id", range(0, len(reds_df)))
    reds_list = [(i, item)  for i, item in enumerate(reds_df["sentence"].to_list())]
    reds_df = reds_df.reset_index(drop=True)
    return reds_list, reds_df


def get_label_df(data_df, mode):
    '''
    get label_df from the data_df, data_df should correspond to the df from the get methods above
    '''
    label_df = pd.DataFrame({
        "statement_id": data_df["statement_id"],
        "statement": data_df["sentence"],
        "care": data_df["care_label"],
        "fairness": data_df["fairness_label"],
        "loyalty": data_df["loyalty_label"],
        "authority": data_df["authority_label"],
        "sanctity": data_df["sanctity_label"],
    })

    if mode in ["reddits", "news", "twitters"]:
        label_df["care_train"] = data_df["care_train"]
        label_df["fairness_train"] = data_df["fairness_train"]
        label_df["authority_train"] = data_df["authority_train"]
        label_df["loyalty_train"] =  data_df["loyalty_train"]
        label_df["sanctity_train"] = data_df["sanctity_train"]

    return label_df


# General stuff

def get_stats(file_pth, mode, dimension = None):
    moral_stat_list, moral_stat_df = get_reddits(file_pth = file_pth, dimension = dimension)

    return moral_stat_list, moral_stat_df

