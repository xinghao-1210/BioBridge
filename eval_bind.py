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
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, ndcg_score

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
    node_types = predictions["node_type"]
    all_tail_types = list(predictions['prediction'].keys())
    
    global_preds = []
    global_nodelabels = []
    global_labels = []
    global_probs = []
    
    ks = [1, 2, 3, 4, 5, 10, 20]
    
    # Iterate over all tail types
    for tail_type in all_tail_types:
        preds = predictions['prediction'][tail_type]  # List of predictions, could be of variable lengths
        pred_probs = predictions['prediction_prob'][tail_type]  # List of predictions, could be of variable lengths
        pred_coss = predictions['prediction_cos'][tail_type]
        labels = predictions['label'][tail_type]  # List of labels, could be of variable lengths
        labels = [l[l != -100] for l in labels]  # Exclude -100 labels
        
        non_empty = [len(label) > 0 for label in labels]  # Mask for non-empty labels
        preds = [pred for i, pred in enumerate(preds) if non_empty[i]]
        pred_probs = [pred_prob for i, pred_prob in enumerate(pred_probs) if non_empty[i]]
        pred_coss = [pred_cos for i, pred_cos in enumerate(pred_coss) if non_empty[i]]
        labels = [label for label in labels if len(label) > 0]
        node_types_subset = [node_types[i] for i in range(len(node_types)) if non_empty[i]]

        # Iterate over samples and compute metrics
        for i, (pred, label, pred_prob,pred_cos) in enumerate(zip(preds, labels, pred_probs,pred_coss)):
            node_type = node_types_subset[i]
            label_set = set(label.tolist())
            
            for k in ks:
                # Handle the case where k is larger than the length of the prediction
                top_k_preds = pred[0, :min(k, len(pred[0]))].tolist()
                pred_set = set(top_k_preds)
                intersection = pred_set.intersection(label_set)
                
                rec = len(intersection) / len(label_set)  # Recall@k
                prec = len(intersection) / k  # Precision@k
                hit = int(len(intersection) > 0)  # Hit@k
                
                metrics[f"head_{node_type}_tail_{tail_type}_rec@{k}"].append(rec)
                metrics[f"head_{node_type}_tail_{tail_type}_prec@{k}"].append(prec)
                metrics[f"head_{node_type}_tail_{tail_type}_hit@{k}"].append(hit)
                
                # Relevance for DCG and iDCG
                relevance = np.isin(top_k_preds, label)
                dcg = np.sum(relevance / np.log2(np.arange(2, min(k, len(relevance)) + 2)))
                metrics[f"head_{node_type}_tail_{tail_type}_dcg@{k}"].append(dcg)
                ideal_relevance = np.sort(relevance)[::-1]
                idcg = np.sum(ideal_relevance / np.log2(np.arange(2, min(k, len(ideal_relevance)) + 2)))
                metrics[f"head_{node_type}_tail_{tail_type}_idcg@{k}"].append(idcg)
                ndcg = dcg / (idcg if idcg > 0 else 1)
                metrics[f"head_{node_type}_tail_{tail_type}_ndcg@{k}"].append(ndcg)

            # Flatten and process predictions for AUC-related metrics
            t_preds = pred[0].tolist()
            t_probs = pred_prob[0].tolist()
            t_coss = pred_cos[0].tolist()
            t_nodelabels = label.tolist()
            t_labels = [1 if l in t_nodelabels else 0 for l in pred[0].tolist()]

            global_preds.extend(t_preds)
            global_probs.extend(t_probs)
            global_nodelabels.extend(t_nodelabels)
            global_labels.extend(t_labels)
            
            # AUC
            auc = roc_auc_score(t_labels, t_probs) if len(set(t_labels))>=2 else 0
            auc_pr = average_precision_score(t_labels, t_probs)
            auc_mroc = roc_auc_score(t_labels, t_probs, multi_class='ovr') if len(set(t_labels))>=2 else 0
            aupr = average_precision_score(t_labels, t_probs, average='macro')
            metrics[f"head_{node_type}_tail_{tail_type}_auc"].append(auc)
            metrics[f"head_{node_type}_tail_{tail_type}_auc-pr"].append(auc_pr)
            metrics[f"head_{node_type}_tail_{tail_type}_auc-mroc"].append(auc_mroc)
            metrics[f"head_{node_type}_tail_{tail_type}_aupr"].append(aupr)
            
            # Precision, Recall, F1 scores
            binary_preds = (np.array(t_probs) >= 0.5).astype(int)
            precision = precision_score(t_labels, binary_preds)
            recall = recall_score(t_labels, binary_preds)
            f1 = f1_score(t_labels, binary_preds)
            metrics[f"head_{node_type}_tail_{tail_type}_prec"].append(precision)
            metrics[f"head_{node_type}_tail_{tail_type}_rec"].append(recall)
            metrics[f"head_{node_type}_tail_{tail_type}_f1"].append(f1)

            # Sort the true labels based on predicted probabilities to calculate overall DCG, IDCG, NDCG
            sorted_indices = np.argsort(t_probs)[::-1]  # Indices of t_probs sorted in descending order
            sorted_labels = np.array(t_labels)[sorted_indices]  # True labels sorted by predicted relevance
            overall_dcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(sorted_labels))
            ideal_sorted_labels = sorted(t_labels, reverse=True)  # Sort true labels in descending order
            overall_idcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_sorted_labels))
            overall_ndcg = overall_dcg / (overall_idcg if overall_idcg > 0 else 1)
            metrics[f"head_{node_type}_tail_{tail_type}_dcg"].append(overall_dcg)
            metrics[f"head_{node_type}_tail_{tail_type}_idcg"].append(overall_idcg)
            metrics[f"head_{node_type}_tail_{tail_type}_ndcg"].append(overall_ndcg)

            # Compute MRR for this sample
            sorted_indices = np.argsort(t_probs)[::-1]  # Sort in descending order
            correct_ranks = [rank for rank, idx in enumerate(sorted_indices) if t_labels[idx] == 1]  # Correct ranks
            if len(correct_ranks) > 0:
                mrr = 1.0 / (np.min(correct_ranks) + 1)
            else:
                mrr = 0.0
            metrics[f"head_{node_type}_tail_{tail_type}_mrr"].append(mrr)

    # Aggregate results across samples for each (head, tail) pair
    new_metrics = {k: np.mean(v) for k, v in metrics.items()}
    # Average over all tail types if more than one
    if len(all_tail_types) > 1:
        avg_metrics = {}
        tail_type_prefixes = set(k.split("_tail_")[0] for k in metrics.keys())
        for prefix in tail_type_prefixes:
            for metric in [f'{d}@{e}' for d in ['rec', 'prec', 'hit', 'dcg', 'idcg', 'ndcg'] for e in ks] + ['mrr', 'prec', 'rec', 'f1', 'auc', 'auc-pr', 'auc-mroc', 'aupr', 'dcg', 'idcg', 'ndcg']:
                tail_metrics = [new_metrics[f"{prefix}_tail_{ttype}_{metric}"] for ttype in all_tail_types if f"{prefix}_tail_{ttype}_{metric}" in new_metrics]
                avg_metrics[f'{prefix}_{metric}_avg'] = np.mean(tail_metrics)
        new_metrics.update(avg_metrics)

    ### Global Evaluation ### 
    global_preds = np.array(global_preds)
    global_nodelabels  = np.array(global_nodelabels)
    global_labels = np.array(global_labels)
    global_probs = np.array(global_probs)
    
    # Compute global AUC, AUC-PR, Precision, Recall, F1
    global_auc = roc_auc_score(global_labels, global_probs) if len(set(global_labels))>=2 else 0
    global_auc_pr = average_precision_score(global_labels, global_probs)
    global_auc_mroc = roc_auc_score(global_labels, global_probs, multi_class='ovr') if len(set(global_labels))>=2 else 0
    global_aupr = average_precision_score(global_labels, global_probs, average='macro')

    global_precision = precision_score(global_labels, (np.array(global_probs) >= 0.5).astype(int))
    global_recall = recall_score(global_labels, (np.array(global_probs) >= 0.5).astype(int))
    global_f1 = f1_score(global_labels, (np.array(global_probs) >= 0.5).astype(int))

    # Compute global MRR
    global_sorted_indices = np.argsort(global_probs)[::-1]
    global_correct_ranks = [rank for rank, idx in enumerate(global_sorted_indices) if global_labels[idx] == 1]
    global_mrr = 1.0 / (np.min(global_correct_ranks) + 1) if len(global_correct_ranks) > 0 else 0.0

    # Compute global NDCG, DCG, and IDCG
    global_relevance = np.isin(np.argsort(global_probs)[::-1], np.argsort(global_labels)[::-1])
    global_dcg = np.sum(global_relevance / np.log2(np.arange(2, len(global_relevance) + 2)))
    ideal_global_relevance = sorted(global_relevance, reverse=True)
    global_idcg = np.sum(ideal_global_relevance / np.log2(np.arange(2, len(ideal_global_relevance) + 2)))
    global_ndcg = global_dcg / (global_idcg if global_idcg > 0 else 1)
                
    # Compute global top-k Hit@K, Precision@K, Recall@K
    for k in ks:
        top_k_global_preds = global_preds[np.argsort(global_probs)[::-1]][:k].tolist()
        top_k_global_set = set(top_k_global_preds)
        global_intersection = top_k_global_set.intersection(set(global_nodelabels.tolist()))
        global_rec_at_k = len(global_intersection) / len(global_nodelabels)
        global_prec_at_k = len(global_intersection) / k
        global_hit_at_k = int(len(global_intersection) > 0)
        new_metrics[f'rec@{k}_global'] = global_rec_at_k
        new_metrics[f'prec@{k}_global'] = global_prec_at_k
        new_metrics[f'hit@{k}_global'] = global_hit_at_k

        # Relevance for DCG and iDCG
        global_k_relevance = np.isin(top_k_global_preds, global_nodelabels)
        global_dcg_at_k = np.sum(global_k_relevance / np.log2(np.arange(2, min(k, len(global_k_relevance)) + 2)))
        global_k_ideal_relevance = np.sort(global_k_relevance)[::-1]
        global_idcg_at_k = np.sum(global_k_ideal_relevance / np.log2(np.arange(2, min(k, len(global_k_ideal_relevance)) + 2)))
        global_ndcg_at_k = global_dcg_at_k / (global_idcg_at_k if global_idcg_at_k > 0 else 1)       
        new_metrics[f'dcg@{k}_global'] = global_dcg_at_k
        new_metrics[f'idcg@{k}_global'] = global_idcg_at_k
        new_metrics[f'ndcg@{k}_global'] = global_ndcg_at_k

    # Add global metrics
    new_metrics["auc_global"] = global_auc
    new_metrics["auc-pr_global"] = global_auc_pr
    new_metrics["auc-mroc_global"] = global_auc_mroc
    new_metrics["aupr_global"] = global_aupr
    new_metrics["precision_global"] = global_precision
    new_metrics["recall_global"] = global_recall
    new_metrics["f1_global"] = global_f1
    new_metrics["mrr_global"] = global_mrr
    new_metrics["dcg_global"] = global_dcg
    new_metrics["idcg_global"] = global_idcg
    new_metrics["ndcg_global"] = global_ndcg
    return new_metrics


def eval_dict2df(eval_dict):
    # Initialize an empty dictionary to hold the transformed data
    transformed_data = {}

    # Process each key-value pair in the original dictionary
    for key, value in eval_dict.items():
        if not key.endswith(('_avg','_global')):
            # Split the key to extract relevant parts
            parts = key.split('_')
            head_tail = '_'.join(parts[1:-1])
            metric = parts[-1]  # rec@ or prec@ or hit@ or dgc@ or mrr
        elif key.endswith('_avg'):
            # Split the key to extract relevant parts
            parts = key.split('_')
            head_tail = '_'.join(parts[:-2])
            metric = '_'.join(parts[-2:])  # avg
        elif key.endswith('_global'):
            head_tail = 'global'
            metric = key.split('eval_')[1]
        else:
            continue

        # Construct a new key for the transformed dictionary
        new_key = head_tail

        # Add the value to the transformed_data dictionary
        if new_key not in transformed_data:
            transformed_data[new_key] = {}
        transformed_data[new_key][metric] = value

    # Convert the transformed_data dictionary to a DataFrame
    eval_df = pd.DataFrame(transformed_data).T
    eval_df = eval_df.sort_index()  # Optionally sort by the index
    return eval_df


# write the data loading module here
def main(
    data_dir="./data/BindData", # the data directory
    split_dir="./data/BindData/train_test_split", # the train/test split directory
    hidden_dim=768, # the hidden dimension of the transformation model
    n_layer=6, # the number of transformer layers
    checkpoint_dir="./checkpoints", # the directory to save the model,
    dataloader_num_workers=8,
    target_relation=2, # the target relation to predict
    target_node_type_index=1, # the index of the target node type
    frequent_threshold=50, # the threshold of the frequent node
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
    val_data = ValDataset(**{"triplet_all":split_data["all"], 
                               "node_test":split_data["node_test"],
                               "node_all":split_data["node_all"],
                               "target_relation": target_relation, # only consider the evaluation on one relation, 2: `interact with`
                               "target_node_type_index": target_node_type_index, # the index of the target node type: protein/gene is 1
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
    # load model from checkpoint_dir
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "model.bin")))
    model.to(device)

    # build trainer
    train_args = TrainingArguments(
        per_device_eval_batch_size=2, # every node corresponds to multiple tail nodes
        dataloader_num_workers=dataloader_num_workers, # number of processes to use for dataloading
        output_dir=None,
        report_to="none",
        )
    
    print("### Training Arguments ###")
    print(json.dumps(train_args.to_dict(), indent=4))

    print("### Number of Trainable Parameters ###")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # build trainer
    trainer = BindingTrainer(
        # args=train_args,
        model=model,
        train_dataset=None,
        eval_dataset=val_data,
        data_collator=TrainCollator(embedding_dict),
        test_data_collator=ValCollator(embedding_dict),
        compute_metrics=compute_metrics,
        )

    # train the model
    eval_dict = trainer.evaluate()
    eval_df = eval_dict2df(eval_dict)
    # display(eval_df)
    eval_df.to_csv(os.path.join(checkpoint_dir, "test_eval.csv"))

    print("### Model Evaluation Done ###")

if __name__ == "__main__":
    fire.Fire(main)