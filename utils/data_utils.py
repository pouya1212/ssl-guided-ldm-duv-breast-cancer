import random
import torch
import pandas as pd
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

def folds_generate(metadata_df, num_folds=5, seed=None):
    if seed is not None:
        random.seed(seed)
    df = metadata_df.copy()
    df['contrast_norm'] = df['Contrast'].str.strip().str.lower()
    contrasts = ['normal', 'dark']
    groups = {}
    for lbl in df['Binary_Label'].unique():
        for ct in contrasts:
            key = (int(lbl), ct)
            bucket = df.loc[(df['Binary_Label'] == lbl) & (df['contrast_norm'] == ct), 'WSI'].tolist()
            random.shuffle(bucket)
            groups[key] = bucket
    folds = [[] for _ in range(num_folds)]
    for key, samples in groups.items():
        n = len(samples)
        per_fold = [n // num_folds + (1 if i < n % num_folds else 0) for i in range(num_folds)]
        idx = 0
        for fold_idx, take in enumerate(per_fold):
            folds[fold_idx].extend(samples[idx: idx+take])
            idx += take
    return folds

def get_loader(args, trainset, valset, testset, all_trainin_validation_set):
    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    train_loader = DataLoader(trainset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4, pin_memory=True)
    val_loader = DataLoader(valset, sampler=SequentialSampler(valset), batch_size=args.eval_batch_size, num_workers=4, pin_memory=True)
    test_loader = DataLoader(testset, sampler=SequentialSampler(testset), batch_size=args.eval_batch_size, num_workers=4, pin_memory=True)
    tune_loader = DataLoader(all_trainin_validation_set, sampler=SequentialSampler(all_trainin_validation_set), batch_size=args.eval_batch_size, num_workers=4, pin_memory=True)
    return train_loader, val_loader, test_loader, tune_loader
