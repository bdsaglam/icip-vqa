# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/09_experiment.ipynb (unless otherwise specified).

__all__ = ['log_training_dataset', 'log_model_evaluation', 'log_preds_for_competition']

# Cell

import numpy as np
from fastai.basics import *

# Cell
def log_training_dataset(df, wandb_enabled=False):
    train_dataframe = df[['video_name', 'frames', 'scene', 'label', 'distortion', 'severity', 'is_valid']]
    path = 'train_dataframe.json'
    train_dataframe.to_json(path, orient='records')
    if wandb_enabled:
        import wandb
        artifact = wandb.Artifact('train_dataframe', type='dataset')
        artifact.add_file(path)
        wandb_run.log_artifact(artifact)
        wandb.log(dict(
            df=wandb.Table(dataframe=train_dataframe),
        ))
    return path

# Cell
import json

def log_model_evaluation(clf_report, scores, wandb_enabled=False):
    clf_report_path = 'classification_report.txt'
    with open(path, 'w') as f:
        f.write(clf_report)

    scores_path = 'scores.json'
    with open(scores_path, 'w') as f:
        json.dump(scores, f)

    if wandb_enabled:
        import wandb

        artifact = wandb.Artifact('classification_report', type='perf')
        artifact.add_file(clf_report_path)
        wandb_run.log_artifact(artifact)

        artifact = wandb.Artifact('scores', type='perf')
        artifact.add_file(scores_path)
        wandb_run.log_artifact(artifact)

        wandb.config.update(scores)

    return clf_report_path, scores_path

# Cell
def log_preds_for_competition(preds, wandb_enabled=False):
    path = 'predict.txt'
    with open(path, 'w') as f:
        f.write('\n'.join(preds))
    if wandb_enabled:
        import wandb
        artifact = wandb.Artifact('test-predictions', type='perf')
        artifact.add_file(path)
        wandb_run.log_artifact(artifact)
    return path