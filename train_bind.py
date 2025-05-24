"""Train the binding model with InfoNCE loss"""
import os
os.environ["WANDB__SERVICE_WAIT"] = "300"

import pdb
from typing import Dict, List, Optional
from collections import defaultdict
import fire
import pickle
import json
import time
import math

# solve the error "too many open files" when data_num_workers > 0
# ref: https://github.com/pytorch/pytorch/issues/11201#issuecomment-421146936
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import torch
import pandas as pd
import numpy as np
from transformers import TrainingArguments
from transformers.trainer_utils import speed_metrics
from transformers.debug_utils import DebugOption
from transformers.trainer_utils import (
    EvalPrediction,
)
from sklearn.metrics import ndcg_score

# source code for the binding model
from src.model import BindingModel
from src.dataset import TrainDataset, ValDataset
from src.collator import TrainCollator, ValCollator
from src.trainer import BindingTrainer
from src.dataset import load_split_data

def build_model_config(data_config):
    # build model config
    model_config = {
        "n_node": len(data_config["node_type"]),
        "n_relation": len(data_config["relation_type"]),
        }
    proj_dim = {}
    for node_type, dim in data_config["emb_dim"].items():
        proj_dim[data_config["node_type"][node_type]] = dim
    model_config["proj_dim"] = proj_dim
    return model_config

def compute_metrics(inputs: EvalPrediction) -> Dict:
    """Compute the metrics for the prediction."""
    metrics = defaultdict(list)
    predictions = inputs.predictions
    num_samples = len(predictions["node_index"])
    node_types = predictions["node_type"]
    all_tail_types = list(predictions['prediction'].keys())
    for tail_type in all_tail_types:
        preds, labels = predictions['prediction'][tail_type], predictions['label'][tail_type]
        for i in range(num_samples):
            # compute r@k, k = 5, 10, 20
            # compute ndcg@k, k = 5
            # recall: tp / (tp+fn)
            node_types_i = node_types[i]
            pred, label = preds[i], labels[i]
            label = label[label!=-100]
            if len(label) > 0:
                # only consider the case where the label is not empty
                for k in [5,10,20]:
                    rec = len(set(pred[0,:k].tolist()).intersection(set(label.tolist()))) / len(label)
                    metrics[f"head_{node_types_i}_tail_{tail_type}_rec@{k}"].append(rec)
                    prec = len(set(pred[0,:k].tolist()).intersection(set(label.tolist()))) / k
                    metrics[f"head_{node_types_i}_tail_{tail_type}_prec@{k}"].append(prec)
                
                # Compute MRR
                mrr = 0.0
                for rank, p in enumerate(pred[0].tolist()):
                    if p in label.tolist():
                        mrr = 1.0 / (rank + 1)
                        break
                metrics[f"head_{node_types_i}_tail_{tail_type}_mrr"].append(mrr)
            
    # compute the sample average
    new_metrics = {}
    for k, v in metrics.items():
        new_metrics[k] = np.mean(v)

    # TODO: average over all tail types if more than one tail type
    if len(all_tail_types) > 1:
        avg_metrics = {}
        tail_type_prefixes = set([k.split("_tail_")[0] for k in metrics.keys()])
        for prefix in tail_type_prefixes:
            tail_metrics = [new_metrics[f"{prefix}_tail_{ttype}_{metric}"] for ttype in all_tail_types 
                            for metric in [f'{d}@{e}' for d in ['rec','prec'] for e in [1,2,3,4,5,10,20]]+['mrr'] 
                            if f"{prefix}_tail_{ttype}_{metric}" in new_metrics]
            avg_metrics[prefix] = np.mean(tail_metrics)

        # Update new_metrics with averaged metrics
        for k, v in avg_metrics.items():
            new_metrics[f"{k}_avg"] = v

    return new_metrics

# write the data loading module here
def main(
    data_dir="./data/BindData", # the data directory
    split_dir="./data/BindData/train_test_split", # the train/test split directory
    hidden_dim=768, # the hidden dimension of the transformation model
    n_layer=6, # the number of transformer layers
    batch_size=1024, # the training batch size
    learning_rate=1.6e-3, # the learning ratesss
    n_epoch=10, # the number of training epochs
    weight_decay=1e-4, # the weight decay
    evaluation_strategy='steps',
    eval_steps=1000, # the number of steps to evaluate the model
    logging_steps=1000,
    save_dir="./checkpoints/model-1", # the directory to save the model
    dataloader_num_workers=4, # the number of workers for data loading
    frequent_threshold=50,
    use_wandb=False, # whether to use wandb
    ):
    # load embedding
    with open(os.path.join(data_dir, "embedding_dict.pkl"), "rb") as f:
        embedding_dict = pickle.load(f)
    
    # load data config
    with open(os.path.join(data_dir, "data_config.json"), "r") as f:
        data_config = json.load(f)

    # load train/test split
    split_data = load_split_data(split_dir)

    # build dataset
    train_data = TrainDataset(**{"triplet":split_data["train"], "node":split_data["node_train"]})
    val_data = ValDataset(**{"triplet_all":split_data["all"], 
                               "node_test":split_data["node_test"],
                               "node_all":split_data["node_all"],
                               "target_relation": None, # only consider the evaluation on one relation, 2: `interact with`
                               "target_node_type_index": None, # the index of the target node type: protein/gene is 1
                               "frequent_threshold": frequent_threshold, # the threshold of the frequent node
                               })

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build the model
    print("### Model Configuration ###")
    # build model config
    model_config = build_model_config(data_config)
    model_config["hidden_dim"] = hidden_dim
    model_config["n_layer"] = n_layer
    print(json.dumps(model_config, indent=4))
    model = BindingModel(**model_config)
    model.to(device)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # save model config to the save directory
    with open(os.path.join(save_dir, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=4)

    # debug
    # debug_train_dataloader(model, train_data, train_collate_fn)
    # debug_test_dataloader(model, val_data, ValCollator(embedding_dict))

    # build trainer
    train_args = TrainingArguments(
        output_dir=save_dir,
        overwrite_output_dir=True,
        num_train_epochs=n_epoch,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=2, # every node corresponds to multiple tail nodes
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        save_steps=100,
        save_total_limit=5,
        do_eval=True,
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        max_grad_norm=1.0, # gradient clipping
        warmup_ratio=0.1,
        dataloader_num_workers=dataloader_num_workers, # number of processes to use for dataloading
        report_to="wandb" if use_wandb else "none",
        )
    
    print("### Training Arguments ###")
    print(json.dumps(train_args.to_dict(), indent=4))

    print("### Number of Trainable Parameters ###")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # build trainer
    trainer = BindingTrainer(
        args=train_args,
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=TrainCollator(embedding_dict),
        test_data_collator=ValCollator(embedding_dict),
        compute_metrics=compute_metrics,
        )

    # train the model
    trainer.train(resume_from_checkpoint=False) # resume_from_checkpoint 

    # save the model    
    trainer.save_model(save_dir)
    torch.save(model.state_dict(), os.path.join(save_dir,"model.bin"))

    print("### Model Saved ###")

if __name__ == "__main__":
    fire.Fire(main)