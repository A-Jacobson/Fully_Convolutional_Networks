import torch
from metrics import binary_accuracy, categorical_accuracy


def test_binary_accuracy():
    y_true = torch.Tensor([0, 1, 0, 1])
    y_preds = torch.Tensor([0.4, 0.6, 0.2, 0.8])
    assert binary_accuracy(y_true, y_preds) == 1.0
    y_true = torch.Tensor([0, 1, 0, 1])
    y_preds = torch.Tensor([1., 0., 1., 0.])
    assert binary_accuracy(y_true, y_preds) == 0


def test_categorical_accuracy():
    y_true = torch.Tensor([0, 1, 0, 1])
    y_pred = torch.Tensor([[0.3, 0.7],
                           [0.6, 0.4],
                           [0.5, 0.5],
                           [0.1, 0.9]])
    assert categorical_accuracy(y_true, y_pred) == 0.5
    y_true = torch.Tensor([0, 1, 0, 1])
    y_pred = torch.Tensor([[1., 0.],
                           [0., 1.],
                           [1., 0.],
                           [0., 1.]])
    assert categorical_accuracy(y_true, y_pred) == 1.0
