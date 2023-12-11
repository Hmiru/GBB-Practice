import torch
from sklearn.metrics import precision_score, recall_score, f1_score
def calculate_metric(true_labels, predicted_labels):
    precision = precision_score(true_labels, predicted_labels, average="macro")
    recall = recall_score(true_labels, predicted_labels, average="macro")
    f1 = f1_score(true_labels, predicted_labels, average="macro")
    return precision, recall, f1

def print_scores(precision, recall, f1):
    print(f"\tPrecision: {precision:.4f}")
    print(f"\tRecall: {recall:.4f}")
    print(f"\tF1 Score: {f1:.4f}")

if __name__=="__main__":
    pass