# experiment.py
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBClassifier

# oversampling/undersampling
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN

# custom modules
import utils
import dataset
import metrics

def run_xgb(X_train, y_train, X_test, scale_pos_weight=None, random_state=42):
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=random_state,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X_train, y_train)
    return model

def _has_majority_neighbors(X, y, minority_label=1, k=5):
    """소수 샘플 중 다수 이웃이 하나도 없는 경우가 존재하는지 빠른 사전 점검."""
    X = np.asarray(X)
    y = np.asarray(y)
    idx_min = np.where(y == minority_label)[0]
    if len(idx_min) == 0 or len(idx_min) == len(y):  # 전부 소수 또는 전부 다수 → ADASYN 부적합
        return False
    nn = NearestNeighbors(n_neighbors=min(k+1, len(y))).fit(X)
    neigh = nn.kneighbors(X[idx_min], return_distance=False)
    # 이웃 리스트에는 자기 자신이 포함되므로 제외
    for row in neigh:
        neighbors = [j for j in row if j != row[0] or len(row)==1]
        if not np.any(y[neighbors] != minority_label):
            return False  # 이 소수 샘플은 다수 이웃이 없음
    return True


def safe_adasyn_fit_resample(X_train, y_train, random_state=42,
                             n_neighbors_grid=(5, 10, 15, 25, 35),
                             sampling_strategy_list=('auto', 0.8, 0.6),
                             fallback='smote'):
    """
    ADASYN 실패 시 n_neighbors/sampling_strategy를 바꿔가며 재시도.
    그래도 안 되면 fallback(SMOTE/BorderlineSMOTE/None).
    """
    # 빠른 예비 체크: k=5에서라도 다수 이웃이 없다면 ADASYN 힘듦
    if not _has_majority_neighbors(X_train, y_train, minority_label=1, k=5):
        if fallback is None:
            raise RuntimeError("ADASYN precheck failed and no fallback set.")
        if fallback == 'smote':
            return SMOTE(random_state=random_state).fit_resample(X_train, y_train), 'SMOTE(fallback)'
        if fallback == 'borderline':
            return BorderlineSMOTE(random_state=random_state).fit_resample(X_train, y_train), 'BorderlineSMOTE(fallback)'
        return (X_train, y_train), 'NO-RESAMPLE'

    # 파라미터 그리드 탐색
    for n_neighbors in n_neighbors_grid:
        for ss in sampling_strategy_list:
            try:
                sampler = ADASYN(random_state=random_state, n_neighbors=n_neighbors, sampling_strategy=ss)
                X_res, y_res = sampler.fit_resample(X_train, y_train)
                return (X_res, y_res), f'ADASYN(n_neighbors={n_neighbors}, ss={ss})'
            except RuntimeError as e:
                # 같은 에러면 다음 설정으로 시도
                if "Not any neigbours belong to the majority class" in str(e):
                    continue
                # 다른 에러는 바로 올려보내기
                raise
    # 폴백
    if fallback == 'smote':
        return SMOTE(random_state=random_state).fit_resample(X_train, y_train), 'SMOTE(fallback)'
    if fallback == 'borderline':
        return BorderlineSMOTE(random_state=random_state).fit_resample(X_train, y_train), 'BorderlineSMOTE(fallback)'
    return (X_train, y_train), 'NO-RESAMPLE'

# ================= Baseline / Sampling =================
def run_baseline_and_sampling(dataset_total, random_state=42):
    utils.seed_everything(random_state)

    methods = {
        "baseline(XGB)": None,
        "RandomOverSampling": RandomOverSampler(random_state=random_state),
        "SMOTE": SMOTE(random_state=random_state),
        "ADASYN": ADASYN(random_state=random_state),
        "BorderlineSMOTE": BorderlineSMOTE(random_state=random_state),
        "RandomUnderSampling": RandomUnderSampler(random_state=random_state),
        "SMOTETomek": SMOTETomek(random_state=random_state),
        "SMOTEENN": SMOTEENN(random_state=random_state),
        "EasyEnsemble": EasyEnsembleClassifier(random_state=random_state),
        "CBLoss": "cbloss",
    }

    for name, sampler in methods.items():
        results = []
        for dataset_id in tqdm(dataset_total, desc=f"{name} datasets", leave=False):
            X, y = dataset.load_and_encode_dataset(dataset_id)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=random_state, stratify=y
            )

            if name == "baseline(XGB)":
                model = run_xgb(X_train, y_train, X_test, random_state=random_state)
            elif name == "CBLoss":
                scale = np.sum(y_train == 0) / max(1, np.sum(y_train == 1))
                model = run_xgb(X_train, y_train, X_test, scale_pos_weight=scale, random_state=random_state)
            elif name == "ADASYN":
                (X_res, y_res), adasyn_mode = safe_adasyn_fit_resample(
                    X_train, y_train,
                    random_state=random_state,
                    n_neighbors_grid=(5,),   # 탐색할 k 값
                    fallback='smote'         # 실패 시 SMOTE로 대체
                )
                model = run_xgb(X_res, y_res, X_test, random_state=random_state)
            elif name == "EasyEnsemble":
                model = EasyEnsembleClassifier(random_state=random_state)
                model.fit(X_train, y_train)
            else:
                X_res, y_res = sampler.fit_resample(X_train, y_train)
                model = run_xgb(X_res, y_res, X_test, random_state=random_state)

            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)

            r_min, r_maj, acc, f1 = metrics.evaluate_performance(
                y_test, y_pred, minority_label=1
            )

            results.append(
                {
                    "name": name,
                    "dataset": dataset_id,
                    "num_train": len(X_train) if name in ["baseline(XGB)", "CBLoss", "EasyEnsemble"] else len(X_res),
                    "recall_minority": r_min,
                    "recall_majority": r_maj,
                    "accuracy": acc,
                    "f1_macro": f1
                }
            )

            print("\nname : ", name)
            print("dataset : ", dataset_id)
            print("recall minority : " , r_min)
            print("recall majority : " , r_maj)
            print("accuracy : " , acc)
            print("f1-macro : " , f1)

        pd.DataFrame(results).to_csv(
            f"./results/{name}_results_{random_state}.csv", index=False
        )


# ================= Main =================
if __name__ == "__main__":

    run_baseline_and_sampling(dataset_total=dataset.TabularDataset.dataset_total, random_state=1)
