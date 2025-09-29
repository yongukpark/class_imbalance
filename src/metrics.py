from sklearn.metrics import recall_score, accuracy_score, f1_score

def evaluate_performance(y_true, y_pred, minority_label=1):
    majority_label = 1 - minority_label
    recall_minority = recall_score(y_true, y_pred, pos_label=minority_label)
    recall_majority = recall_score(y_true, y_pred, pos_label=majority_label)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')  # <- macro f1
    return recall_minority, recall_majority, acc, f1