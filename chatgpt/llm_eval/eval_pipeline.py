import pandas as pd
import argparse
from sklearn.metrics import roc_curve, auc, f1_score
import matplotlib.pyplot as plt
from scripts.llm_eval.llm_utils import data_process
import ast
import json

def get_precision_f1_auc(result_file, label_df, threshold, mode, is_everything):
    '''
    Label DF should have the same structure as score_df
    '''
    prob_df = pd.read_csv(result_file)

    score_df_trunc = prob_df.iloc[:, 2:]
    predicted = (score_df_trunc > threshold).astype(int)

    # Create a new DataFrame with the invariant column statement_id, statement
    raw_prob_df = prob_df[["statement_id", "statement"]]
    prob_df = prob_df.sort_values(by="statement_id")

    predicted_df = pd.concat([raw_prob_df, predicted], axis=1)
    predicted_df = predicted_df.sort_values(by="statement_id")

    roc_dict = {}
    precision = {}
    f1_score_dict = {}
    print(predicted_df["statement_id"],label_df["statement_id"])

    for col in prob_df.iloc[:, 2:].columns:
        # get rid of the rows with invalid labels
        if is_everything == False and mode in ["reddits", "news", "twitters"]:
            f1_to_drop_ids = label_df.loc[(label_df[col] == -1) | (label_df[f"{col}_train"] == 1) , "statement_id"]
        else:
            f1_to_drop_ids = label_df.loc[label_df[col] == -1, "statement_id"]

        if mode in ["reddits", "news", "twitters"]:
            to_drop_ids = label_df.loc[(label_df[col] == -1) | (label_df[f"{col}_train"] == 1) , "statement_id"]
        else:
            to_drop_ids = label_df.loc[label_df[col] == -1, "statement_id"]
        trunc_label_df = label_df[~label_df["statement_id"].isin(to_drop_ids)]
        trunc_predicted_df = predicted_df[~predicted_df["statement_id"].isin(to_drop_ids)]
        f1_trunc_label_df = label_df[~label_df["statement_id"].isin(f1_to_drop_ids)]
        f1_trunc_predicted_df = predicted_df[~predicted_df["statement_id"].isin(f1_to_drop_ids)]
        trunc_prob_df = prob_df[~prob_df["statement_id"].isin(to_drop_ids)]
        print(trunc_predicted_df.shape, trunc_label_df.shape)

        # Calculate f1 score
        f1_score_dict[col] = f1_score(f1_trunc_label_df[col].tolist(), f1_trunc_predicted_df[col].tolist())

        # Calculate precision for each column
        true_positives = ((trunc_predicted_df[col] == 1) & (trunc_label_df[col] == 1)).sum()
        predicted_positives = (trunc_predicted_df[col] == 1).sum()
        precision[col] = true_positives / predicted_positives if predicted_positives > 0 else 0
        # Calculate ROC curve for each dimension
        fpr, tpr, thresholds = roc_curve(trunc_label_df[col].to_list(), trunc_prob_df[col].to_list()) 
        roc_dict[col] = (fpr, tpr, auc(fpr, tpr))

    return precision, f1_score_dict, roc_dict


def plot_ROC(fpr, tpr, roc_auc, output_file, name):
    plt.figure()  
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {name}')
    plt.legend()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_file",
        type=str,
        default="output/news/reason_first/gpt4omini/gpt4omini_scores.csv",
        help="data to run moral dimension extraction on",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/mf_corpora_merged.csv",
        help="the original dataset",
    )
    parser.add_argument(
        "--out_folder",
        type=str,
        default="output/news/reason_first/gpt4omini"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="news"
    )
    parser.add_argument(
        "--is_everything",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--verbose", type=bool, default=True, help="whether to print out results"
    )

    args = parser.parse_args()

    _ , ms_df = data_process.get_stats(args.data, args.mode)
    
    if args.mode != "mfq":
        label_df = data_process.get_label_df(ms_df, mode=args.mode)
    else:
        label_df = ms_df

    precisions, f1_result, roc_dict = get_precision_f1_auc(args.result_file, label_df=label_df, threshold=0.5, mode = args.mode, is_everything = args.is_everything)
    data = {
        "precision": precisions,
        "f1": f1_result
    }
    with open(args.out_folder + "/result.json", 'w') as f:
        json.dump(data, f)
    for key, roc_tup in roc_dict.items():
        plot_ROC(roc_tup[0], roc_tup[1], roc_tup[2], args.out_folder + '/' + key + ".png", key)
    

if __name__ == "__main__":
    main()