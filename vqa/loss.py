# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/04_loss.ipynb (unless otherwise specified).

__all__ = ['CombinedLoss']

# Cell
from fastai.basics import *

# Cell

class CombinedLoss():
    def __init__(self, *loss_funcs, weight=None):
        if weight is None:
            weight = [1.]*len(loss_funcs)
        self.weight = weight
        self.loss_funcs = loss_funcs

    def __call__(self, outs, *targets, **kwargs):
        return sum([
            w*loss_func(out, target)
            for loss_func, w, out, target in zip(self.loss_funcs, self.weight, outs, targets)
        ])

    def activation(self, outs):
        return [getattr(loss_func, 'activation', noop)(out) for loss_func, out in zip(self.loss_funcs, outs)]

    def decodes(self, outs):
        return [getattr(loss_func, 'decodes', noop)(out) for loss_func, out in zip(self.loss_funcs, outs)]
