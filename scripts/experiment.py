import warnings
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
from vqa.inference import *


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

arch_map = dict(resnet18=resnet18, resnet50=resnet50, resnet101=resnet101)
model_map = dict(baseline_mtm=BaselineMTM, multiscale_mtm=MultiScaleMTM, sequence_mtm=SequenceMTM)
def make_model(dls, arch_name, model_name, pretrained):
    arch = arch_map[arch_name]
    model = model_map[model_name](arch=arch, n_distortion=len(dls.vocab[0]), n_sev=len(dls.vocab[1]), pretrained=pretrained).to(DEVICE)
    return model

loss_map = dict(ce=CrossEntropyLossFlat, focal=FocalLossFlat)
def make_loss(distortion_loss_name, severity_loss_name, loss_weights):
    return CombinedLoss(loss_map[distortion_loss_name](), loss_map[severity_loss_name](), weight=loss_weights)
    
def train_eval_infer(
    df,
    tst_df,
    *,
    normalize: bool = False,
    augment: bool = False,
    bs: int = 16, 
    arch_name: str = 'resnet50', 
    model_name: str = 'baseline_mtm', 
    pretrained: bool = True, 
    distortion_loss_name: str = 'ce', 
    severity_loss_name: str = 'ce', 
    loss_weights: list=(1.0, 1.0),
    fine_tune: bool=True,
    freeze_epochs: int=10,
    epochs: int=40,
    lr: float = None,
    wandb_enabled=True,
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
        blocks=(ImageTupleBlock, CategoryBlock, CategoryBlock),
        splitter=ColSplitter('is_valid'),
        get_x=ColReader('frame_paths'),
        get_y=[ColReader('distortion'), ColReader('severity')],
        batch_tfms=batch_tfms,
        n_inp=1
    )
    dls = db.dataloaders(df, bs=bs)
    model = make_model(dls, arch_name=arch_name, model_name=model_name, pretrained=pretrained)

    # metrics
    distortion_f1_macro = route_to_metric(0, F1Score(average='macro'))
    distortion_f1_macro.name = 'distortion_f1_macro'
    distortion_accuracy = route_to_metric(0, accuracy)
    distortion_accuracy.func.__name__ = 'distortion_accuracy'
    severity_f1_macro = route_to_metric(1, F1Score(average='macro'))
    severity_f1_macro.name = 'severity_f1_macro'
    severity_accuracy = route_to_metric(1, accuracy)
    severity_accuracy.func.__name__ = 'severity_accuracy'

    # callbacks
    cbs=[
        SaveModelCallback(),
        EarlyStoppingCallback(patience=10),
    ]
    if wandb_enabled:
        cbs.append(WandbCallback())
    
    # learner
    learn = Learner(
        dls, 
        model=model,
        loss_func= make_loss(distortion_loss_name, severity_loss_name, loss_weights),
        metrics=[distortion_f1_macro, distortion_accuracy, severity_f1_macro, severity_accuracy],
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
    clf_report, scores = evaluate_mtl(dls.vocab, probs, targets, preds)
    if wandb_enabled:
        with open('classification_report.txt', 'w') as f:
            f.write(clf_report)
        artifact = wandb.Artifact('classification_report', type='perf')
        artifact.add_file('classification_report.txt')
        wandb_run.log_artifact(artifact)
        wandb.config.update(scores)
    
    # inference
    inference_df = get_test_inferences(dls, learn, tst_df)
    lines = make_submission(inference_df['distortion_inference'], inference_df['severity_inference'])
    if wandb_enabled:
        artifact = wandb.Artifact('test-predictions', type='perf')
        artifact.add_file('predict.txt')
        wandb_run.log_artifact(artifact)
        
    return dls, learn


def prepare_train_dataframe(df, directory, frame_indices_list, drop_reference):
    df['video_path'] = df['video_name'].apply(lambda vn: str(Path(directory) / vn))
    return (
        df
        .pipe(lambda dataf: dataf[dataf.label != 'R_0'])
        .pipe(make_framer(frame_indices_list))
        .pipe(remove_corrupt_video_frames)
    )

def make_dataframes(train_dataframe_path, train_dir, tst_dir, frame_indices_list, drop_reference):
    df = prepare_train_dataframe(pd.read_json(train_dataframe_path), train_dir, frame_indices_list, drop_reference)
    tst_df = make_test_dataframe(Path(tst_dir), frame_indices_list=frame_indices_list)
    assert_stratied_split(df, 'label')
    return df, tst_df

def setup_experiment():
    import os
    from pathlib import Path
    from datetime import datetime
    experiment_id = datetime.now().isoformat().rsplit('.', 1)[0].replace(':', '-')
    experiment_dir = Path('./experiments') / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(experiment_dir)

def resolve_paths(config):
    for field_path in ['data.train_dataframe_path', 'data.train_dir', 'data.tst_dir']:
        fields = field_path.split('.')
        node = config
        for field in fields[:-1]:
            node = node[field]
        node[fields[-1]] = str(Path(node[fields[-1]]).resolve())
    return config

def run_experiment(config):
    seed = config.get('seed')
    if seed is not None:
        set_seed(seed)

    # wandb
    wandb_enabled = config['wandb'].get('wandb_enabled', True)
    if wandb_enabled:
        wandb_run = wandb.init(
            project=config['wandb']['wandb_project'], 
            entity=config['wandb']['wandb_username']
        )
        wandb.config.update(flatten_dict(config))

    # data
    df, tst_df = make_dataframes(**config['data'])
    print(len(df), L(df.distortion.unique().tolist()), L(df.severity.unique().tolist()))
    assert len(df.distortion.unique()) == 18
    assert len(df.severity.unique()) == 4

    # experiment
    setup_experiment()
    dls, learn = train_eval_infer(
        df,
        tst_df,
        **config['model'],
        wandb_enabled=wandb_enabled,
    )

    # log dataset
    train_dataframe = df[['video_name', 'frames', 'scene', 'label', 'distortion', 'severity', 'is_valid']]
    train_dataframe.to_json('train_dataframe.json', orient='records')
    if wandb_enabled:
        artifact = wandb.Artifact('train_dataframe', type='dataset')
        artifact.add_file('train_dataframe.json')
        wandb_run.log_artifact(artifact)
        wandb.log(dict(
            df=wandb.Table(dataframe=train_dataframe),
        ))

    # wrap up
    if wandb_enabled:
        wandb.finish()


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg')
    args = parser.parse_args()
    
    with open(args.cfg) as f:
        config = json.load(f)
    resolve_paths(config)
    
    run_experiment(config)
    