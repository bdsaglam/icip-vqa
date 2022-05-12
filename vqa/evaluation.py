# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/07_evaluation.ipynb (unless otherwise specified).

__all__ = ['evaluate_mtl', 'evaluate_stl']

# Cell

import warnings
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from fastai.metrics import F1Score, accuracy
import matplotlib.pyplot as plt
from fastai.basics import *

# Cell
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning, module=r'.*')

def evaluate_mtl(vocabs, probs, targets, preds, show=False):
    clf_reports = []
    for vocab, target, pred in zip(vocabs, targets, preds):
        target = target.cpu().numpy()
        pred = pred.cpu().numpy()
        label_indices = list(range(len(vocab)))
        clf_report = classification_report(target, pred, labels=label_indices, target_names=[*vocab])
        clf_reports.append(clf_report)
        if show:
            fig, ax = plt.subplots(figsize=(16, 12))
            ConfusionMatrixDisplay.from_predictions(target, pred, labels=label_indices, display_labels=vocab, ax=ax)
    scores = dict(
        distortion_f1_macro = F1Score(average='macro')(preds[0], targets[0]).item(),
        distortion_accuracy = accuracy(probs[0], targets[0]).item(),
        severity_f1_macro = F1Score(average='macro')(preds[1], targets[1]).item(),
        severity_accuracy = accuracy(probs[1], targets[1]).item(),
    )
    return '\n'.join(clf_reports), scores

# Cell

warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning, module=r'.*')

def evaluate_stl(vocab, prob, target, pred, show=False):
    clf_report = classification_report(
        target.cpu().numpy(),
        pred.cpu().numpy(),
        labels=list(range(len(vocab))),
        target_names=vocab
    )
    if show:
        fig, ax = plt.subplots(figsize=(16, 12))
        ConfusionMatrixDisplay.from_predictions(target, pred, labels=label_indices, display_labels=vocab, ax=ax)
    scores = dict(
        f1_macro = F1Score(average='macro')(pred, target).item(),
        accuracy = accuracy(prob, target).item(),
    )
    return clf_report, scores