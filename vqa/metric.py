# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/05_metric.ipynb (unless otherwise specified).

__all__ = ['LearnerProxy', 'metric_route']

# Cell

from types import MethodType
from fastcore.basics import GetAttr
from fastai.metrics import Metric, AvgMetric


class LearnerProxy(GetAttr):
    _default = 'learn'
    def __init__(self, learn, idx):
        self.learn = learn
        self.idx = idx
        self.pred = self.learn.pred[idx]
        self.y = self.learn.y[idx]


def metric_route(idx, metric):
    if isinstance(metric, type):
        metric = metric()
    if not isinstance(metric, Metric):
        func = lambda preds, *targs, **kwargs: metric(preds[idx], targs[idx], **kwargs)
        func.__name__ = metric.__name__
        return AvgMetric(func)
    accumulate = metric.accumulate
    metric.accumulate = MethodType(lambda self, learn: accumulate(LearnerProxy(learn, idx)), metric)
    call = metric.__call__
    metric.__call__ = MethodType(lambda self, preds, targs: call(preds[idx], targs[idx]), metric)
    return metric