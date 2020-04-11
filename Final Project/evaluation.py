
from sklearn import metrics
import numpy as np

def evaluate(y_true, y_pred, pos_label=1):
    """
    Computes performance metrics for binary predicted and true outcomes.
    :param y_true: (n x 1) vector of true outcomes. Entries must be 0 or 1.
    :param y_pred: (n x 1) vector of predicted outcomes. Entries must be 0 or 1.
    :return: (dict) key: metric name, val: score
    """

    assert np.all(np.isin(y_true, [0, 1]))
    assert np.all(np.isin(y_pred, [0, 1]))

    result = {"accuracy": metrics.accuracy_score(y_true, y_pred),
              "auc": metrics.roc_auc_score(y_true, y_pred),
              "f1_score": metrics.f1_score(y_true, y_pred),
              "precision": metrics.precision_score(y_true, y_pred, pos_label=pos_label),
              "recall": metrics.recall_score(y_true, y_pred, pos_label=pos_label),
              }

    return result


def test_evaluate_binary_preds():
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])

    m = evaluate(y_true, y_pred)

    assert m['accuracy'] == 0.9
    assert m['auc'] == 0.9
    assert np.isclose(m['f1_score'],  8 / 9)
    assert m['precision'] == 1.0
    assert m['recall'] == 0.8


def test_evaluate():
    test_evaluate_binary_preds()

    print("All tests passing.")


if __name__ == "__main__":
    test_evaluate()