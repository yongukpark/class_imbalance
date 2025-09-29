# experiment.py

# =============== basic modules ===============
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# =============== graph construction ===============
import networkx as nx
from sklearn.neighbors import NearestNeighbors

# =============== classifier ===============
from xgboost import XGBClassifier

# =============== import custom modules ===============
import utils
import dataset
import metrics


def make_hop_weights(hops, alpha=0.55):
    raw = {h: (alpha ** (h-1)) for h in hops}
    s = sum(raw.values())
    return {h: (w / s) for h, w in raw.items()} if s > 0 else {h: 1/len(hops) for h in hops}


def run_experiment(random_state=42, minority_label=1, max_iter=100, alpha=0.55):
    utils.seed_everything(random_state)
    results = []

    for dataset_id in tqdm(dataset.TabularDataset.dataset_total, desc="proposed_model", leave=False):
        X, y = dataset.load_and_encode_dataset(dataset_id)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )

        X_train = X_train.values if hasattr(X_train, "values") else X_train
        y_train = y_train.values if hasattr(y_train, "values") else y_train
        X_test = X_test.values if hasattr(X_test, "values") else X_test
        y_test = y_test.values if hasattr(y_test, "values") else y_test

        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)

        knn = NearestNeighbors(n_neighbors=6, algorithm='kd_tree', n_jobs=-1, metric='euclidean')
        knn.fit(X_train)
        neighbors = knn.kneighbors(X_train, return_distance=False)

        G = nx.Graph()
        num_nodes = X_train.shape[0]
        G.add_nodes_from(range(num_nodes))

        for i in range(num_nodes):
            for j in neighbors[i]:
                if i != j:
                    G.add_edge(i, j)

        minority_idx = np.where(y_train == minority_label)[0]
        included_nodes = set(minority_idx)

        hop_nodes_dict = {0: set(minority_idx)}

        for hop in range(1, max_iter + 1):
            hop_neighbors = set()
            for node in hop_nodes_dict[hop-1]:
                neighbors = nx.single_source_shortest_path_length(G, node, cutoff=1)
                hop_neighbors.update([n for n in neighbors if n != node])

            new_nodes = hop_neighbors - included_nodes
            if len(new_nodes) == 0:
                break

            included_nodes.update(new_nodes)
            hop_nodes_dict[hop] = new_nodes

        available_hops = sorted([h for h in hop_nodes_dict.keys() if h >= 1])
        hop_models = {}
        for h in available_hops:
            train_nodes = set(hop_nodes_dict[0]) | set(hop_nodes_dict[h])
            train_idx = list(train_nodes)
            X_sub, y_sub = X_train[train_idx], y_train[train_idx]

            clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state)
            clf.fit(X_sub, y_sub)
            hop_models[h] = clf

        weights = make_hop_weights(available_hops, alpha=alpha)

        proba_sum = None
        for h, clf in hop_models.items():
            w = weights[h]
            proba_h = clf.predict_proba(X_test)
            proba_sum = (proba_h * w) if proba_sum is None else (proba_sum + proba_h * w)

        ensemble_pred = np.argmax(proba_sum, axis=1)

        r_min, r_maj, acc, f1 = metrics.evaluate_performance(y_test, ensemble_pred, minority_label)

        results.append({
            "name": "Proposed",
            "dataset": f"{dataset_id}",
            "num_train": len(X_train),
            "recall_minority": r_min,
            "recall_majority": r_maj,
            "accuracy": acc,
            "f1_macro": f1,
        })
        print("\ndataset : ", dataset_id)
        print("recall minority : " , r_min)
        print("recall majority : " , r_maj)
        print("accuracy : " , acc)
        print("f1-macro : " , f1)


    df = pd.DataFrame(results)
    df.to_csv(f"./results/proposed_results_{random_state}.csv", index=False)


if __name__ == "__main__":
    results = run_experiment(random_state=1, alpha=0.55)
