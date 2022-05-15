import warnings
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastai.vision.all import *
from fastai.callback.wandb import *
from sklearn.exceptions import UndefinedMetricWarning
import wandb

from vqa.utils import *
from vqa.wrangling import *
from vqa.datablock import *
from vqa.model import *
from vqa.loss import *
from vqa.metric import *
from vqa.evaluation import *
from vqa.inference_stl import *
from vqa.experiment import *


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

arch_map = dict(resnet18=resnet18, resnet50=resnet50, resnet101=resnet101)
model_map = dict(baseline_stm=BaselineSTM, sequence_stm=SequenceSTM)
def make_model(dls, arch_name, model_name, pretrained):
    arch = arch_map[arch_name]
    model = model_map[model_name](arch=arch, n_out=len(dls.vocab), pretrained=pretrained).to(DEVICE)
    return model


loss_map = dict(ce=CrossEntropyLossFlat, focal=FocalLossFlat)


def train_eval_infer(
    df,
    tst_df,
    *,
    normalize: bool = False,
    augment: bool = False,
    bs: int = 16, 
    arch_name: str = 'resnet50', 
    model_name: str = 'baseline_stm', 
    pretrained: bool = True, 
    loss_name: str = 'ce', 
    fine_tune: bool=True,
    freeze_epochs: int=10,
    epochs: int=40,
    lr: float = None,
    wandb_run=None,
):
    batch_tfms = []
    if normalize:
        batch_tfms.append(Normalize.from_stats(*imagenet_stats))
    if augment:
        batch_tfms.extend(
            setup_aug_tfms([
                Dihedral(p=1., draw=lambda x: torch.from_numpy(np.array(random.choices([0,1,2,4], k=x.size(0)))).to(x.device)),
            ])
        )
    db = DataBlock(
        blocks=(ImageTupleBlock, CategoryBlock),
        splitter=ColSplitter('is_valid'),
        get_x=ColReader('frame_paths'),
        get_y=ColReader('label'),
        batch_tfms=batch_tfms,
    )
    dls = db.dataloaders(df, bs=bs)
    model = make_model(dls, arch_name=arch_name, model_name=model_name, pretrained=pretrained)

    # callbacks
    cbs=[
        SaveModelCallback(),
        EarlyStoppingCallback(patience=10),
    ]
    if wandb_run:
        cbs.append(WandbCallback())
    
    # learner
    learn = Learner(
        dls, 
        model=model,
        loss_func=loss_map[loss_name](),
        metrics=[F1Score(average='macro'), accuracy],
        splitter=model.splitter,
        cbs=cbs,
    )
    if DEVICE.type!='cpu': 
        learn = learn.to_fp16()

    if lr is None:
        lr_res = learn.lr_find(start_lr=1e-6, end_lr=1e-1, num_it=200)
        lr = lr_res.valley
    
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning, module=r'.*')
        if fine_tune:
            learn.fine_tune(epochs, lr, freeze_epochs=freeze_epochs)
        else:
            learn.fit_one_cycle(epochs, lr)

    # evaluation
    learn = learn.load('model')
    probs, targets, preds = learn.get_preds(dl=dls.valid, with_decoded=True)
    clf_report, scores = evaluate_stl(dls.vocab, probs, targets, preds)
    log_model_evaluation(clf_report, scores, wandb_run)
    
    # inference
    inference_df = get_test_inferences(dls, learn, tst_df)
    lines = make_submission_preds(inference_df['pred'].values)
    log_preds_for_competition(lines, wandb_run)
    
    return dls, learn


def run_experiment(config):
    seed = config.get('seed')
    if seed is not None:
        set_seed(seed)
    # wandb
    wandb_run = None
    if config.get('wandb', dict()).get('wandb_enabled', False):
        wandb_run = wandb.init(
            project=config['wandb']['wandb_project'], 
            entity=config['wandb']['wandb_username']
        )
        wandb.config.update(flatten_dict(config))
    # data
    df, tst_df = make_dataframes(**config['data'])
    assert_stratied_split(df, 'label')
    print(len(df), L(df.label.unique().tolist()))
    # experiment
    dls, learn = train_eval_infer(
        df,
        tst_df,
        **config['model'],
        wandb_run=wandb_run,
    )
    # log dataset
    log_training_dataset(df, wandb_run)
    # wrap up
    if wandb_run:
        wandb.finish()
    return df, tst_df, dls, learn


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg')
    args = parser.parse_args()
    
    with open(args.cfg) as f:
        config = json.load(f)
    
    for field_path in ['data.train_dataframe_path', 'data.train_dir', 'data.tst_dir']:
        resolve_path(config, field_path)
    
    with set_dir(make_experiment_dir()):
        run_experiment(config)
    
