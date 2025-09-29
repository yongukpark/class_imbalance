import numpy as np
import pandas as pd

import dataset

baseline = "baseline(XGB)"
models = ["baseline(XGB)","RandomOverSampling","SMOTE","ADASYN","BorderlineSMOTE",
          "RandomUnderSampling", "SMOTETomek", "SMOTEENN", "EasyEnsemble", "CBLoss", "Proposed"]

path = './results/'

def select_features(model, seeds, dataset_list):
    all_ratio_recall = []
    all_ratio_f1 = []

    for seed in seeds:
        base_file = f"{path}{baseline}_results_{seed}.csv"
        model_file = f"{path}{model}_results_{seed}.csv"

        df_baseline = pd.read_csv(base_file)
        df = pd.read_csv(model_file)

        for dataset_id in dataset_list:
            # ===== recall =====
            base_rec_series = df_baseline.loc[df_baseline['dataset'] == dataset_id, 'recall_minority']
            comp_rec_series = df.loc[df['dataset'] == dataset_id, 'recall_minority']
            if base_rec_series.empty or comp_rec_series.empty:
                continue
            ratio_recall = comp_rec_series.iloc[0] / base_rec_series.iloc[0]
            all_ratio_recall.append(ratio_recall)

            # ===== f1 macro =====
            base_f1_series = df_baseline.loc[df_baseline['dataset'] == dataset_id, 'f1_macro']
            comp_f1_series = df.loc[df['dataset'] == dataset_id, 'f1_macro']
            if base_f1_series.empty or comp_f1_series.empty:
                continue
            ratio_f1 = comp_f1_series.iloc[0] / base_f1_series.iloc[0]
            all_ratio_f1.append(ratio_f1)

    mean_recall = np.mean(all_ratio_recall)
    std_recall  = np.std(all_ratio_recall, ddof=1) if len(all_ratio_recall) > 1 else 0.0
    mean_f1 = np.mean(all_ratio_f1)
    std_f1  = np.std(all_ratio_f1, ddof=1) if len(all_ratio_f1) > 1 else 0.0

    print(f"{model:<25} "
          f"{mean_recall:.4f} ({std_recall:.4f})  "
          f"{mean_f1:.4f} ({std_f1:.4f})")

    return mean_recall, mean_f1


if __name__ == "__main__":
    random_state_list = [1]

    print(f"\n📈 Metric")
    print(f"{'model':<25} {'mean(r)':>10} {'std(r)':>10} {'mean(f)':>10} {'std(f)':>10}")

    for model in models: 
        select_features(model, random_state_list, dataset.TabularDataset.dataset_total)